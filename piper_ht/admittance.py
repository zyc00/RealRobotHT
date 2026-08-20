"""Admittance control on top of the Piper's position loop.

The arm has no usable torque mode (JointMitCtrl is inert on firmware S-V1.9-0),
so it cannot be made mechanically backdrivable.  Instead we measure how hard the
position loop is fighting an external push and walk the *setpoint* toward it.

    residual = measured_effort - fitted_gravity(q) - zero
    tau_user = -residual
    qdot     = deadband(tau_user) / damping
    q_cmd   += qdot * dt

Everything is in SDK reported-effort units, matching piper_ht.gravity_fit.

Sign is measured, not assumed: in data/drag_test.npz the arm deflects INTO a
push, giving a negative correlation between deflection and residual on every
joint with measurable deflection (J2 -0.43, J3 -0.29, J4 -0.20).  Hence
tau_user = -residual.
"""

import time

import numpy as np

from .arm import JOINT_LIMITS
from .gravity_fit import GravityFit, MIN_DEADBAND


class AdmittanceController:
    def __init__(self, arm, gravity=None,
                 enabled=(True, True, True, True, True, True),
                 deadband=None,
                 damping=(6.0, 7.0, 6.0, 6.0, 4.0, 2.5),
                 mass=0.0,
                 hysteresis=0.35,
                 v_max=0.8,
                 leash_deg=6.0,
                 max_travel_deg=90.0,
                 zero_tc=15.0,
                 speed_pct=55):
        self.arm = arm
        self.grav = gravity or GravityFit()
        # Optional learned free-space model; when present it replaces the
        # analytic gravity prediction entirely.
        self.next_est = None
        # Speed-dependent deadband, fitted from NEXT's validation error:
        #     rms_err(v) = a + b*v,  v = max|qdot| over all joints
        # Model error grows with arm speed, so a single worst-case constant
        # would make slow dragging needlessly heavy.
        self.db_a = None
        self.db_b = None
        self.db_k = 3.0
        self.enabled = np.array(enabled, bool)
        # Default deadband is the measured noise floor with a little headroom.
        self.deadband = np.array(deadband if deadband is not None
                                 else MIN_DEADBAND * 1.15, float)
        self.damping = np.array(damping, float)
        # Breakaway vs. keep-moving threshold.  Starting a drag has to clear the
        # full deadband, but once a joint is moving we only need to stay above
        # `hysteresis` of it - otherwise every joint feels as heavy to keep
        # moving as it did to start, which is what made this feel sticky.
        self.hysteresis = float(hysteresis)
        self.active = np.zeros(6, bool)
        # Virtual inertia.  0 means the massless first-order law qdot = tau/D,
        # which is the lightest response for a given D.  Positive M makes the
        # arm resist changes in velocity - it does NOT make dragging lighter -
        # but it low-passes the velocity with time constant M/D, which is what
        # lets you lower D past the point where torque noise makes it jitter.
        self.mass = np.asarray(mass, float) * np.ones(6)
        self.qdot = np.zeros(6)
        # Oscillation guard: a joint that keeps reversing is in a limit cycle,
        # not being dragged.  Latching 'active' made this self-sustaining, so we
        # count reversals and force a cooldown.
        self._sign_hist = [[] for _ in range(6)]
        self._cooldown = np.zeros(6)
        self.max_reversals = 6          # per second
        self.cooldown_s = 1.0
        # Drift guard: a human push is bursty, but model bias pushes forever.
        # A joint held above its deadband continuously for this long is almost
        # certainly being driven by model error, so freeze it and re-zero.
        self.max_sustained_s = 8.0
        self._active_since = np.full(6, np.inf)
        self.v_max = v_max
        self.leash = np.radians(leash_deg)
        self.max_travel = np.radians(max_travel_deg)
        self.zero_tc = zero_tc
        self.speed_pct = speed_pct
        self.zero = np.zeros(6)
        self.q_start = None
        self.q_cmd = None

    def predict_free_effort(self, q):
        """Effort the arm should draw with nothing touching it."""
        if self.next_est is not None:
            return self.next_est.push(q, self.arm.dq(),
                                      self.q_cmd if self.q_cmd is not None else q)
        return self.grav.effort(q)

    def residual(self, q=None):
        q = self.arm.q() if q is None else q
        return self.arm.effort() - self.predict_free_effort(q)

    def capture_zero(self, seconds=2.0, hold=None):
        hold = self.arm.q() if hold is None else hold
        Z = []
        t0 = time.time()
        while time.time() - t0 < seconds:
            self.arm.move_j(hold, speed_pct=10)
            Z.append(self.residual())
            time.sleep(0.01)
        Z = np.array(Z)
        self.zero = Z.mean(0)
        return self.zero, Z.std(0)

    def attach_next(self, estimator, deadband_path="data/next_deadband.npz"):
        """Use a learned free-space model, with its speed-dependent deadband."""
        self.next_est = estimator
        try:
            z = np.load(deadband_path)
            self.db_a, self.db_b = z["a"], z["b"]
            self.deadband = self.db_k * self.db_a
        except Exception:
            pass
        return self

    def current_deadband(self, v):
        if self.db_a is None:
            return self.deadband
        return self.db_k * (self.db_a + self.db_b * v)

    def start(self):
        self.q_start = self.arm.q()
        self.q_cmd = self.q_start.copy()

    def step(self, dt):
        q = self.arm.q()

        # A model that predicts an impossible torque must never drive the arm.
        if self.next_est is None and not self.grav.is_plausible(q):
            self.arm.move_j(self.q_cmd, speed_pct=self.speed_pct)
            return q, np.zeros(6), np.zeros(6), "gravity model implausible at this pose"

        res = self.residual(q) - self.zero
        tau_user = -res

        # Schmitt trigger: latch a joint 'active' once it clears the full
        # deadband, and hold it active until it falls below the lower one.
        amag = np.abs(tau_user)
        db = self.current_deadband(float(np.abs(self.arm.dq()).max()))
        self.active = np.where(self.active, amag > db * self.hysteresis, amag > db)
        db_eff = np.where(self.active, db * self.hysteresis, db)
        mag = np.maximum(amag - db_eff, 0.0)
        tau_eff = np.sign(tau_user) * mag

        heavy = self.mass > 1e-9
        qdot = np.where(heavy, self.qdot, 0.0)
        # M*qddot + D*qdot = tau_eff, integrated where M > 0.
        if heavy.any():
            acc = (tau_eff - self.damping * self.qdot) / np.where(heavy, self.mass, 1.0)
            self.qdot = np.where(heavy, self.qdot + acc * dt, 0.0)
            qdot = np.where(heavy, self.qdot, 0.0)
        qdot = np.where(heavy, qdot, tau_eff / self.damping)
        qdot = np.clip(qdot, -self.v_max, self.v_max)
        qdot[~self.enabled] = 0.0

        # Count direction reversals over a 1 s window; freeze any joint that is
        # chattering rather than tracking a hand.
        now = time.time()
        for i in range(6):
            h = self._sign_hist[i]
            if abs(qdot[i]) > 0.02:
                if h and np.sign(qdot[i]) != h[-1][1]:
                    h.append((now, np.sign(qdot[i])))
                elif not h:
                    h.append((now, np.sign(qdot[i])))
            self._sign_hist[i] = [x for x in h if now - x[0] < 1.0]
            if len(self._sign_hist[i]) > self.max_reversals:
                self._cooldown[i] = now + self.cooldown_s
                self._sign_hist[i] = []
                self.active[i] = False
        # Sustained one-sided drive -> treat as drift, not a push.
        for i in range(6):
            if self.active[i] and abs(qdot[i]) > 0.02:
                if not np.isfinite(self._active_since[i]):
                    self._active_since[i] = now
                elif now - self._active_since[i] > self.max_sustained_s:
                    self._cooldown[i] = now + self.cooldown_s
                    self.active[i] = False
                    self._active_since[i] = np.inf
                    self.zero[i] += res[i]      # absorb the offset that caused it
            else:
                self._active_since[i] = np.inf

        frozen = self._cooldown > now
        qdot[frozen] = 0.0
        self.qdot = np.where(heavy, qdot, 0.0)
        self.qdot[frozen] = 0.0

        # Re-zero slowly, and only while nothing looks like a push, so that
        # leftover model error does not accumulate into a drift.
        quiet = ~self.active
        if self.zero_tc > 0:
            self.zero[quiet] += (dt / self.zero_tc) * res[quiet]

        self.q_cmd = self.q_cmd + qdot * dt
        self.q_cmd = np.clip(self.q_cmd, q - self.leash, q + self.leash)
        self.q_cmd = np.clip(self.q_cmd, JOINT_LIMITS[:, 0] + np.radians(2),
                             JOINT_LIMITS[:, 1] - np.radians(2))
        self.q_cmd[~self.enabled] = self.q_start[~self.enabled]

        stop = None
        if np.abs(q - self.q_start).max() > self.max_travel:
            stop = "travel limit (%.0f deg) reached" % np.degrees(self.max_travel)

        self.arm.move_j(self.q_cmd, speed_pct=self.speed_pct)
        return q, tau_user, qdot, stop
