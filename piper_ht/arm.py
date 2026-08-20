"""Thin extension of piperx_teleop.arm for this repo's research scripts.

The core wrapper now lives in the piperx_teleop package; only the extras the
research scripts still need are kept here - MIT torque control (inert on this
firmware, see scripts/03_mit_probe.py), raw motor current, and the standby/stop
paths used during bring-up.
"""

import numpy as np
from piperx_teleop.arm import CMD2RAD, JOINT_LIMITS, RAD2CMD  # noqa: F401
from piperx_teleop.arm import PiperArm as _PiperArm

# JointMitCtrl encodes t_ref into this range regardless of what the SDK
# docstring claims. Moot in practice: MIT does nothing on firmware S-V1.9-0.
MIT_TORQUE_LIMIT = 8.0


class PiperArm(_PiperArm):
    def connect(self, settle=0.3, require_control=False):
        # Research scripts read state in teach mode deliberately, so this does
        # not raise by default the way the teleop package's does.
        return super().connect(settle=settle, require_control=require_control)

    def current(self):
        """Raw motor current (A)."""
        return np.array([m.current for m in self._motors()]) * 1e-3

    def state(self):
        import time
        return dict(t=time.time(), q=self.q(), dq=self.dq(),
                    effort=self.effort(), current=self.current())

    def mit_mode(self):
        """Enter MIT mode. Inert on firmware S-V1.9-0 - see 03_mit_probe.py."""
        self.piper.MotionCtrl_2(0x01, 0x01, 0, 0xAD)

    def mit(self, i, pos, vel, kp, kd, tau):
        tau = float(np.clip(tau, -MIT_TORQUE_LIMIT, MIT_TORQUE_LIMIT))
        self.piper.JointMitCtrl(i + 1, float(pos), float(vel), float(kp), float(kd), tau)

    def standby(self):
        self.piper.MotionCtrl_2(0x00, 0x00, 0, 0x00)

    def stop(self):
        self.piper.MotionCtrl_1(0x01, 0, 0)
