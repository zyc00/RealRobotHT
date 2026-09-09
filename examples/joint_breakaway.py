"""B601 extension: independent breakaway fractions and pre-cap telemetry.

The archived reference core remains unchanged. update() is its recurrence with
only scalar breakaway replaced by six fractions and output-cap telemetry added.
Uniform fractions are regression-tested against the reference.
"""
import numpy as np

from b601_reference_balance import BalancedDrag


class JointBreakawayDrag(BalancedDrag):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.break_fractions = np.full(self.n, self.break_beta)
        self.breakaway_torque = np.zeros(self.n)
        self.pre_cap = np.zeros(self.n)
        self.saturated = np.zeros(self.n, dtype=bool)

    def set_break(self, beta):
        if not np.isfinite(beta) or not 0 <= beta <= 1:
            raise ValueError('breakaway fraction must be 0..1')
        self.break_beta = float(beta)
        self.break_fractions[:] = beta
        return self.break_beta

    def set_joint_break(self, joint, beta):
        if not 0 <= joint < self.n or not np.isfinite(beta) or not 0 <= beta <= 1:
            raise ValueError('invalid joint breakaway fraction')
        self.break_fractions[joint] = beta
        return float(beta)

    def update(self, q, v, dt=0.01) -> np.ndarray:
        self.lam_d += (self._lam_goal - self.lam_d) * (dt / (0.5 + dt))
        self._observe(q, v, dt)
        if self._idle >= 2:                # fresh DRAG stretch -> ramp the output back in
            self._ramp_t = 0.0
            self._q_latch = None           # re-latch the detent at the new starting pose
        self._idle = 0
        self._ramp_t += dt

        # Jacobian + cond(J) once (shared by shaping, Cartesian damping, and alpha-kappa schedule)
        J = self.dyn.ee_jacobian(q, frame="local")
        ev_j = np.linalg.eigvalsh(J @ J.T)
        cond = float(np.sqrt(max(ev_j.max(), 1e-12) / max(ev_j.min(), 1e-12)))
        r_db = np.sign(self.r) * np.maximum(np.abs(self.r) - self.r0, 0.0)   # dead-banded hand estimate

        # (4) alpha scheduling: taper friction comp near zero velocity (avoid stiction chatter) and
        #     near singularities (avoid amplifying uncertain friction). 0 -> disabled (=1).
        s_v = 1.0 if self.alpha_sigma_v <= 0 else (1.0 - np.exp(-(v / self.alpha_sigma_v) ** 2))
        s_k = 1.0 if self.alpha_kappa0 <= 0 else float(np.clip(self.alpha_kappa0 / max(cond, 1e-6), 0.0, 1.0))

        # friction feed-forward, paper-style (eq 13): applied whenever the joint moves, direction
        # and taper from tanh(v/v0) alone - NO drive gate, NO sustain margins (relief is inherently
        # "sustained"). Passivity against calibration error is guaranteed by the Cartesian damping
        # D_v below (eq 37) + kd_drag + the fric_scale headroom: a worst-case over-relief Delta
        # injects at most Delta*v^2/v0 of power near rest, so joint damping >= Delta/v0 dissipates it.
        # (6) hybrid soft intent gate: full paper relief while the hand clearly drives a joint,
        # floor*relief when it does not (motor creep, bias, coasting) - see the constructor note.
        gate = np.clip(self.r * np.sign(v) / self.r0, 0.0, 1.0)
        s_int = self.gate_floor + (1.0 - self.gate_floor) * gate
        lvl = self.fric_curve(v) if self.fric_on else np.zeros(self.n)
        f_ff = np.tanh(v / self.v0) * lvl * s_int
        f_ff = f_ff * s_v * s_k
        # (1) viscous friction comp: cancel B*qd (paper eq 13). Self-zeroing at rest, so ungated -
        # but SATURATED at the identification range (the fv sweep only measured |qd| <= ~0.2 rad/s).
        # This term is anti-damping (+0.85*B*qd, positive velocity feedback); extrapolating it
        # linearly ran j6 (B 0.112 vs kd 0.05: net slope +0.045*qd) away to vel_abort at 4 rad/s.
        # Saturating caps the injection at 0.85*B*visc_vsat (~0.03 N.m) - a feel term, not a driver.
        if self.fric_on and self.visc_on and np.any(self.fric_viscous):
            vv = self.visc_vsat
            f_ff = f_ff + self.fric_scale * self.fric_viscous * vv * np.tanh(v / vv) * s_int * s_k
        # (3) breakaway assist: when the hand is pushing (r_db != 0) pre-pay a fraction of the
        #     static breakaway in that direction, decaying as the joint gets moving. Direction
        #     from sign(r_db) - no F/T sensor needed. Helps proximal joints break free.
        self.breakaway_torque[:] = 0.
        if self.fric_on and np.any(self.break_fractions > 0.0):
            # direction-dependent breakaway (paper eq 28): the hand's intended direction sign(r_db)
            # selects which threshold to pre-pay - cheap gearboxes break away asymmetrically
            raw_dir = np.where(r_db >= 0.0, self._raw_static_pos, self._raw_static_neg)
            f_static = np.minimum(raw_dir + self._raw_static_mu * self._g_abs, self.level_max)
            decay = np.exp(-np.abs(v) / max(self.break_vs, 1e-3))
            self.breakaway_torque = self.break_fractions * self.fric_scale * f_static * np.sign(r_db) * decay * s_k
            f_ff = f_ff + self.breakaway_torque

        r_net = np.clip(r_db + f_ff, -self.r_net_max, self.r_net_max)

        # (2) Cartesian virtual damping D_v (passivity-aware): tau = -J^T D_v J qd. Strictly
        #     dissipative (qd . tau <= 0), so it can dominate friction-estimate error and keep
        #     the rendered interaction passive - THE guard for the ungated relief (eq 37).
        #     Two robustness deviations from the raw J^T D_v J: (a) DIAGONAL only - the
        #     off-diagonal terms torque *other* joints, and under worst-case over-relief each
        #     recruited joint gets its own relief, a positive-feedback channel (seen in sim);
        #     the diagonal keeps eq-37's Cartesian sizing while staying per-joint dissipative.
        #     (b) discrete-time cap: DELAYED damping (one cycle + the velocity-filter lag)
        #     destabilizes a joint once d*dt/M_jj approaches ~0.3 (light wrists first; seen on
        #     hardware as a wrist limit cycle whose amplitude is bounded by the saturation
        #     ceiling d_eff*vsat - raising the knee raised the lid). Cap at 0.15*M_jj/dt.
        #     (c) SATURATING, not linear: the guard is only needed near zero velocity (over-relief
        #     injection is capped at Delta*tanh(v/v0), so its danger zone is slow creep). A linear
        #     damper sized for that (~1 N.m.s/rad) costs ~1 N.m of drag at guiding speed - the arm
        #     feels rigid. tau = -d_eff * vsat * tanh(v/vsat): slope d_eff at rest (guard intact -
        #     dissipation >= injection for all v when vsat >= v0), felt drag capped at d_eff*vsat
        #     (~0.15-0.2 N.m). 'damp_vs <rad/s>' tunes the knee live.
        tau_damp = np.zeros(self.n)
        if self.damp_t > 0.0 or self.damp_r > 0.0:
            Dv = np.diag([self.damp_t] * 3 + [self.damp_r] * 3)
            Dj = J.T @ (Dv @ J)
            cap = 0.15 * np.maximum(np.diag(self._M), 1e-6) / max(dt, 1e-3)
            vs = self.damp_vsat
            tau_damp = -np.minimum(np.diag(Dj), cap) * vs * np.tanh(v / vs)

        # (5) latched detent: hold the pose whenever the hand is NOT driving the joint. The latch
        # follows q while you push (r_db != 0) and freezes when you release, so the spring holds
        # against an un-driven drift (the near-vertical lean) at any speed, but never fights guiding.
        # Gating on the hand estimate r (not velocity) is what lets it arrest a bias that would
        # otherwise accelerate past a velocity threshold.
        tau_detent = np.zeros(self.n)
        if self.detent_kp > 0.0:
            if self._q_latch is None:
                self._q_latch = q.copy()
            driven = np.abs(r_db) > 0.0
            self._q_latch = np.where(driven, q, self._q_latch)          # follow the hand; freeze when released
            tau_detent = self.detent_kp * (~driven) * (self._q_latch - q)

        self._runaway_check(v, r_db, dt)
        if self.shaping_on:
            K = self._shaping_matrix(J)
            sing = np.clip((40.0 - cond) / 15.0, 0.0, 1.0)  # fade shaping between cond(J) 25 and 40
            self.alpha = min(1.0, self._ramp_t / self.ramp_s) * sing * self._trip_scale
            self.pre_cap = self.alpha * (K @ r_net) + f_ff + tau_damp + tau_detent
        else:
            self.alpha = 0.0
            self.pre_cap = f_ff + tau_damp + tau_detent
        self.saturated = np.abs(self.pre_cap) > self.tau_cap
        self.tau_out = np.clip(self.pre_cap, -self.tau_cap, self.tau_cap)
        return self.tau_out
