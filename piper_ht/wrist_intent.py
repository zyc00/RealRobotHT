"""Project wrist (J4-6) torques onto the base joints (J1-3) to get the direction
the operator's push is trying to move each base joint.

Model: the operator applies a force f at a grasp point p on the gripper (a
"push, don't twist" prior).  Every joint i feels tau_i = c_i . f with
c_i = a_i x (p - o_i)  (a_i joint axis, o_i a point on it, all in the base
frame).  Stack the wrist rows as A^T (3x3) and the base rows as B^T (3x3):

    tau_w = A^T f            (rank 2: the tool-axis component of f is invisible)
    f_hat = (A^T)^+ tau_w    (truncated pseudo-inverse; tool-axis part -> 0)
    tau_b = B^T f_hat        (the torque that push applies to J1-3)

sign(tau_b) is the direction to apply stiction/kinetic-friction compensation.
The whole thing is one 3x3 matrix per pose, M(q) = B^T (A^T)^+, so the
direction is sign(M(q) tau_w).

Use the torque mapping, not the kinematic one (q123_dot = J123^+ J456 dq):
stiction breakaway follows the applied joint torque, and the two disagree in
sign at most poses.

    from piper_ht.wrist_intent import WristIntent
    wi = WristIntent(model, grasp=(0, 0, 0.15))     # grasp point, gripper_base frame, m
    s, tau_b = wi.direction(q, tau_wrist)           # s in {-1, 0, +1}^3
    tau_comp = s * ratio * stiction                 # your calibrated per-joint numbers

tau_wrist must be the *external* wrist torque: measured joint torque minus the
gravity model (model.gravity_torque(q)[3:]) and, if you hold the wrist with an
impedance, minus the controller's own contribution.  If you only have a wrist
deflection dq under a diagonal stiffness K, use tau_wrist = K @ dq.
"""

import numpy as np

_LINKS = ["link1", "link2", "link3", "link4", "link5", "link6"]


class WristIntent:
    def __init__(self, model, grasp=(0.0, 0.0, 0.15), ee_link="gripper_base",
                 rank_tol=1e-3, gate=0.15, tau_min=0.0):
        """
        model     PiperModel (piper_ht.model or piperx_teleop)
        grasp     where the hand holds, in the ee_link frame (m).  On the J6 axis
                  for a normal gripper grip; a lateral offset barely helps.
        rank_tol  singular values below rank_tol * s_max are treated as zero
                  (this is what drops the unobservable tool-axis component)
        gate      a base joint gets direction 0 unless |tau_b_i| exceeds
                  gate * max_j |tau_b_j|  (flips live in the small-torque joints)
        tau_min   |tau_wrist| below this is treated as no push (sensor noise)
        """
        self.m, self.grasp, self.ee = model, np.asarray(grasp, float), ee_link
        self.rank_tol, self.gate, self.tau_min = rank_tol, gate, tau_min

    # ------------------------------------------------------------------ geometry
    def columns(self, q):
        """c_i = a_i x (p - o_i) for the six joints, and the grasp point p."""
        q = np.asarray(q, float)
        T = self.m.fk(q, link=self.ee)
        p = T[:3, :3] @ self.grasp + T[:3, 3]
        C = np.empty((6, 3))
        for i, ln in enumerate(_LINKS):
            Ti = self.m.fk(q, link=ln)          # link frame: z = joint axis, origin on it
            C[i] = np.cross(Ti[:3, 2], p - Ti[:3, 3])
        return C, p

    def projection(self, q):
        """M(q) = B^T (A^T)^+  (3x3): wrist torques -> base torques.  Also A^T, B^T."""
        C, _ = self.columns(q)
        At, Bt = C[3:], C[:3]                    # tau = At f  /  Bt f
        U, s, Vt = np.linalg.svd(At)
        keep = s > self.rank_tol * s[0]
        At_pinv = (Vt[keep].T / s[keep]) @ U[:, keep].T
        return Bt @ At_pinv, At, Bt

    # ------------------------------------------------------------------ estimate
    def base_torque(self, q, tau_wrist):
        """Estimated torque the operator's push applies to J1-3, and f_hat."""
        M, At, Bt = self.projection(q)
        tau_wrist = np.asarray(tau_wrist, float)
        f_hat = np.linalg.pinv(At, rcond=self.rank_tol) @ tau_wrist
        return M @ tau_wrist, f_hat

    def direction(self, q, tau_wrist):
        """Per-base-joint direction in {-1, 0, +1} plus the underlying tau_b."""
        tau_wrist = np.asarray(tau_wrist, float)
        if np.linalg.norm(tau_wrist) < self.tau_min:
            return np.zeros(3, int), np.zeros(3)
        tau_b, _ = self.base_torque(q, tau_wrist)
        big = np.abs(tau_b) > self.gate * np.abs(tau_b).max()
        return (np.sign(tau_b) * big).astype(int), tau_b


if __name__ == "__main__":
    # self-check against a finite-difference Jacobian of the grasp point
    from piper_ht.model import PiperModel
    m = PiperModel()
    wi = WristIntent(m)
    rng = np.random.default_rng(0)
    lo = np.array([-2.6, 0, -2.97, -1.55, -1.55, -2.09]); hi = -lo; hi[1] = 3.14; hi[2] = 0
    worst = 0.0
    for _ in range(50):
        q = rng.uniform(lo, hi)
        C, p0 = wi.columns(q)
        def probe(qq):
            T = m.fk(qq, link=wi.ee); return T[:3, :3] @ wi.grasp + T[:3, 3]
        for i in range(6):
            d = np.zeros(6); d[i] = 1e-6
            fd = (probe(q + d) - probe(q - d)) / 2e-6
            worst = max(worst, np.abs(fd - C[i]).max())
    print(f"max |analytic - finite-difference| Jacobian column error: {worst:.2e} m/rad")
    s, tb = wi.direction(np.zeros(6), [1.0, 0.0, 0.0])
    print("q=0, tau_wrist=+1 N.m on J4  ->  direction", s, " tau_b", np.round(tb, 3))
    print("M(0) =\n", np.round(wi.projection(np.zeros(6))[0], 3))
