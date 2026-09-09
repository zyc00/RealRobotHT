"""B601 reference control core on Piper's TorqueSession; no scheduled bias or Piper observer.

Core snapshot: b601_teleop/b601/balance.py at working tree 2026-09-08,
HEAD df021c4b616fe7a7c50708dd3421cb1d877e7d93 (working tree may differ).
SHA256 888329aa0fee08c9b062dc5f895763397337e86d001f779065f6ef4dbbeac471.
"""
import argparse
import json
from pathlib import Path
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

from b601_reference_balance import BalancedDrag
from piperx_teleop import MitCommand, TorqueSession, model_with_tool


class VelocityEstimator:
    """Same position-difference / 30 ms low-pass recurrence as B601 arm.py."""
    def __init__(self, n=6, tau_s=0.03):
        self._prev = None
        self.v = np.zeros(n)
        self.tau = tau_s

    def update(self, q, dt):
        if self._prev is None:
            self._prev = q.copy()
            return self.v
        raw = (q - self._prev) / max(dt, 1e-4)
        self._prev = q.copy()
        a = dt / (self.tau + dt)
        self.v += a * (raw - self.v)
        return self.v


class DynamicsAdapter:
    """Piper local z is roll; express its Jacobian in B601's roll-x convention.

    Apply the same proper cyclic rotation to linear and angular components.
    Joint coordinates/torques and M/C/g are unchanged.
    """
    def __init__(self, dyn):
        self.base = dyn
        self.nq = dyn.nq

    def mass_matrix(self, q):
        return self.base.mass_matrix(q)

    def coriolis(self, q, v):
        return self.base.coriolis(q, v)

    def gravity(self, q):
        return self.base.gravity(q)

    def ee_jacobian(self, q, frame='local'):
        if frame != 'local':
            raise ValueError('reference drag uses the local tool frame')
        return self.base.jacobian(q, frame)[[2, 0, 1, 5, 3, 4], :]


class ReferenceLaw:
    def __init__(self, model, core, kd, tau_max, hz=100):
        self.model, self.core = model, core
        self.kd, self.tau_max = np.asarray(kd, float), np.asarray(tau_max, float)
        self.hz = hz
        self.velocity = VelocityEstimator()
        self.previous_t = None
        self.rows = []
        self.setting_rows = []
        self.configuration = {}
        self.log_enabled = False
        self.lock = threading.Lock()
        self.telem = {}

    def __call__(self, s):
        with self.lock:
            dt = 1 / self.hz if self.previous_t is None else s.t - self.previous_t
            if not np.isfinite(dt) or dt <= 0:
                raise ValueError('invalid control timestamp')
            self.previous_t = s.t
            v = self.velocity.update(s.q, dt)
            g = self.model.gravity_torque(s.q)
            extra = self.core.update(s.q, v, dt)
            ff = np.clip(g + extra, -self.tau_max, self.tau_max)
            zero = np.zeros(6)
            # Same B601 command policy: firmware damping, zero stiffness,
            # current position as target; record post-limit feed-forward.
            self.core.note_sent(ff, zero, self.kd, s.q, zero)
            self.telem = dict(q=s.q.tolist(), velocity=v.tolist(), gravity=g.tolist(),
                              torque_ff=ff.tolist(), residual=self.core.r.tolist(),
                              balance_total=extra.tolist(), alpha=float(self.core.alpha),
                              trips=int(self.core.trips))
            if self.log_enabled:
                self.rows.append(np.concatenate([[s.t], s.q, v, g, ff, self.core.r,
                                                  extra, self.kd, [self.core.alpha, self.core.trips]]))
                c = self.core
                self.setting_rows.append([c.kappa, c.fric_scale, c.damp_t, c.damp_r,
                                          c.break_beta, c.break_vs])
            return MitCommand(t_ff=ff, p_des=s.q.copy(), v_des=zero, kp=zero, kd=self.kd.copy())

    def state(self):
        with self.lock:
            c = self.core
            return dict(self.telem, kappa=c.kappa, fric_scale=c.fric_scale,
                        damp_t=c.damp_t, damp_r=c.damp_r, break_beta=c.break_beta,
                        break_vs=c.break_vs, kd=self.kd.tolist(), mode='b601-reference')

    def set(self, name, value):
        ranges = dict(kappa=(0, 2), fric_scale=(0, 0.9), damp_t=(0, 20), damp_r=(0, 0.3),
                      break_beta=(0, 1), break_vs=(0.005, 0.3))
        if name not in ranges or not np.isfinite(value) or not ranges[name][0] <= value <= ranges[name][1]:
            raise ValueError('invalid parameter or value')
        with self.lock:
            if name == 'kappa':
                self.core.set_kappa(value)
            elif name == 'fric_scale':
                self.core.set_fric_scale(value)
            else:
                setattr(self.core, name, value)
        return value

    def save_log(self, path):
        np.savez(path, data=np.asarray(self.rows), columns=np.array(
            't q(6) velocity(6) gravity(6) torque_ff(6) residual(6) balance_total(6) firmware_kd(6) alpha trips'.split()),
            controller='b601-reference', final_settings=json.dumps(self.state()),
            configuration=json.dumps(self.configuration), settings=np.asarray(self.setting_rows),
            settings_columns=np.array('kappa fric_scale damp_t damp_r break_beta break_vs'.split()),
            core_sha256='888329aa0fee08c9b062dc5f895763397337e86d001f779065f6ef4dbbeac471')


def serve(law, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            u = urlparse(self.path)
            status, content_type = 200, 'application/json'
            try:
                if u.path == '/':
                    body = Path(__file__).with_name('b601_panel.html').read_bytes()
                    content_type = 'text/html; charset=utf-8'
                elif u.path == '/state':
                    body = json.dumps(law.state()).encode()
                elif u.path == '/set':
                    qs = parse_qs(u.query)
                    value = law.set(qs.get('p', [''])[0], float(qs.get('v', ['nan'])[0]))
                    body = json.dumps(dict(ok=True, value=value)).encode()
                else:
                    status, body = 404, b'{}'
            except ValueError as exc:
                status, body = 400, json.dumps(dict(ok=False, error=str(exc))).encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--controller', choices=['b601'], default='b601', help=argparse.SUPPRESS)
    ap.add_argument('--can', default='can0')
    ap.add_argument('--tool', required=True)
    ap.add_argument('--gravity', required=True, help='uses constant legacy scale/bias only')
    ap.add_argument('--friction', required=True)
    ap.add_argument('--balance', type=float, default=0.0)
    ap.add_argument('--fric-scale', type=float, default=0.0)
    ap.add_argument('--balance-md', type=float, default=1.8)
    ap.add_argument('--balance-irot', type=float, default=0.06)
    ap.add_argument('--balance-iroll', type=float, default=0.0005)
    ap.add_argument('--balance-fo', type=float, default=3.0)
    ap.add_argument('--balance-gate-floor', type=float, default=0.5)
    ap.add_argument('--fric-sigma-v', type=float, default=0.03)
    ap.add_argument('--fric-kappa0', type=float, default=0.0)
    ap.add_argument('--damp', type=float, nargs=2, default=(2.0, 0.3))
    ap.add_argument('--damp-vsat', type=float, default=0.08)
    ap.add_argument('--balance-breakaway', type=float, default=0.0,
                    help='B601 scalar assist, observer-directed on all joints; default off')
    ap.add_argument('--breakaway-vs', type=float, default=0.1)
    ap.add_argument('--kd', type=float, nargs=6, default=[0.0]*6,
                    help='raw firmware MIT damping gains (physical units uncalibrated on Piper)')
    ap.add_argument('--rate', type=float, default=100.0)
    ap.add_argument('--tcp', type=float, default=0.19)
    ap.add_argument('--duration', type=float, default=0)
    ap.add_argument('--yes', action='store_true')
    ap.add_argument('--serve', type=int, nargs='?', const=8731)
    ap.add_argument('--log')
    ap.add_argument('--dry-run', action='store_true', help='validate models without connecting to CAN')
    a = ap.parse_args(argv)
    values = [a.balance, a.fric_scale, a.balance_md, a.balance_irot, a.balance_iroll,
              a.balance_fo, a.balance_gate_floor, a.fric_sigma_v, a.fric_kappa0,
              *a.damp, a.damp_vsat, a.balance_breakaway, a.breakaway_vs, *a.kd, a.rate, a.tcp, a.duration]
    if not np.isfinite(values).all() or min(values) < 0:
        ap.error('all numeric settings must be finite and nonnegative')
    if not (0 <= a.balance <= 2 and 0 <= a.fric_scale <= 0.9 and 0 <= a.balance_gate_floor <= 1
            and 0 <= a.balance_breakaway <= 1 and 20 <= a.rate <= 200 and max(a.kd) <= 0.5
            and min(a.balance_md, a.balance_irot, a.balance_iroll, a.balance_fo, a.breakaway_vs) > 0
            and 0.08 <= a.damp_vsat <= 0.5):
        ap.error('invalid gain/rate/target; firmware kd limited to 0..0.5 for this experimental port')
    for path in (a.tool, a.gravity, a.friction):
        if not Path(path).is_file():
            ap.error('missing calibration: ' + path)
    with np.load(a.gravity) as data:
        for key in ('scale', 'bias'):
            if data[key].shape != (6,) or not np.isfinite(data[key]).all():
                ap.error('invalid gravity ' + key)
    with np.load(a.friction) as data:
        mapping = dict(fric='kin_f0', f_static='static_f0', f_static_pos='static_pos',
                       f_static_neg='static_neg', fric_mu='kin_mu', f_static_mu='static_mu', fric_viscous='kin_B')
        params = {k: np.asarray(data[v], float).copy() for k, v in mapping.items()}
    if any(v.shape != (6,) or not np.isfinite(v).all() or np.any(v < 0) for v in params.values()):
        ap.error('invalid friction coefficients')
    from piperx_teleop.dynamics import ArmDynamics
    # Same URDF-only inertia policy as B601 (no Piper-only rotor augmentation).
    dyn = DynamicsAdapter(ArmDynamics(tool=a.tool, tcp_offset=a.tcp, gravity=a.gravity, rotor=0))
    model = model_with_tool(a.tool, gravity=a.gravity)
    core = BalancedDrag(dyn, kappa=a.balance, fric_scale=a.fric_scale, **params,
                        m_d=a.balance_md, i_rot=a.balance_irot, i_roll=a.balance_iroll, f_o=a.balance_fo,
                        gate_floor=a.balance_gate_floor, alpha_sigma_v=a.fric_sigma_v,
                        alpha_kappa0=a.fric_kappa0, damp_t=a.damp[0], damp_r=a.damp[1],
                        damp_vsat=a.damp_vsat, break_beta=a.balance_breakaway, break_vs=a.breakaway_vs,
                        tau_cap=np.array([0.6, 1.5, 1.0, 0.4, 0.3, 0.2]))
    core.set_kappa(a.balance)
    law = ReferenceLaw(model, core, a.kd, [8, 10, 8, 3, 3, 3], a.rate)
    law.configuration = vars(a).copy()
    law.log_enabled = a.log is not None
    print('B601 reference core; constant gravity; 30 ms velocity filter; no extra residual filter.')
    print('Gravity uses the file\'s legacy scale/bias; static/motion scheduling is disabled.')
    print('B601 gate floor', a.balance_gate_floor, 'velocity taper', a.fric_sigma_v,
          'Piper-specific torque caps; firmware kd', a.kd)
    print('Firmware kd units on Piper are uncalibrated: matching command fields does not prove matching damping.')
    if a.dry_run:
        q = np.zeros(6)
        np.testing.assert_allclose(model.gravity_torque(q), dyn.gravity(q), atol=1e-8)
        print('Model validation passed; no CAN connection made.')
        return
    if a.log and Path(a.log).exists():
        ap.error('log exists; choose a new --log path')
    session = TorqueSession(law, can=a.can, hz=a.rate)
    server = serve(law, a.serve) if a.serve is not None else None
    if server:
        print('B601 panel: http://127.0.0.1:%d' % a.serve)
    try:
        if not a.yes:
            input('>>> ENTER to go compliant ')
        session.run(duration=a.duration)
    finally:
        if server:
            server.shutdown()
        if a.log:
            law.save_log(a.log)


if __name__ == '__main__':
    main()
