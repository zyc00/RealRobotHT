"""Thin, unit-safe wrapper around C_PiperInterface_V2.

Everything crossing this boundary is SI: radians, rad/s, N.m, metres.
"""

import time

import numpy as np
from piper_sdk import C_PiperInterface_V2

RAD2CMD = 1000.0 * 180.0 / np.pi     # rad -> SDK 0.001 deg
CMD2RAD = 1.0 / RAD2CMD

# URDF joint limits, used as a clamp on every outgoing command.
JOINT_LIMITS = np.array([
    [-2.618, 2.618],
    [0.0,    3.14],
    [-2.967, 0.0],
    [-1.745, 1.745],
    [-1.22,  1.22],
    [-2.0944, 2.0944],
])

# t_ref is encoded into this range by the SDK regardless of what the public
# docstring claims (see C_PiperInterface_V2.__JointMitCtrl defaults).
MIT_TORQUE_LIMIT = 8.0


class PiperArm:
    def __init__(self, can_name="can0"):
        self.piper = C_PiperInterface_V2(can_name)
        self._connected = False

    # ---------- lifecycle ----------

    def connect(self, settle=0.2):
        self.piper.ConnectPort()
        time.sleep(settle)
        self._connected = True
        return self

    def enable(self, timeout=5.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.piper.EnablePiper():
                return True
            time.sleep(0.01)
        return all(self.piper.GetArmEnableStatus())

    def disable(self):
        self.piper.DisableArm(7)

    def is_enabled(self):
        return list(self.piper.GetArmEnableStatus())

    # ---------- state ----------

    def q(self):
        j = self.piper.GetArmJointMsgs().joint_state
        return np.array([j.joint_1, j.joint_2, j.joint_3,
                         j.joint_4, j.joint_5, j.joint_6]) * CMD2RAD

    def _motors(self):
        h = self.piper.GetArmHighSpdInfoMsgs()
        return [h.motor_1, h.motor_2, h.motor_3, h.motor_4, h.motor_5, h.motor_6]

    def dq(self):
        """Joint velocity (rad/s). Reported as int16, so this one is signed."""
        return np.array([m.motor_speed for m in self._motors()]) * 1e-3

    def current(self):
        """Raw motor current (A) as reported by the driver."""
        return np.array([m.current for m in self._motors()]) * 1e-3

    def effort(self):
        """Joint torque magnitude (N.m), current * the SDK's fixed coefficient."""
        return np.array([m.effort for m in self._motors()]) * 1e-3

    def gripper(self):
        g = self.piper.GetArmGripperMsgs().gripper_state
        return g.grippers_angle * 1e-6, g.grippers_effort * 1e-3

    def control_mode(self):
        return str(self.piper.GetArmStatus().arm_status.ctrl_mode)

    def in_teach_mode(self):
        """True when the arm is in drag-teach mode and ignoring CAN commands.

        Teaching mode is entered with the button on the arm, and the firmware
        will NOT leave it on request: MotionCtrl_1 with grag_teach_ctrl 0x00,
        0x02 and 0x06, and track_ctrl 0x04, were all tested and ignored.  The
        only ways out are the button again, a power cycle, or ResetPiper (which
        drops the arm).  While in this mode every JointCtrl/EndPoseCtrl is
        silently discarded, which looks exactly like broken motion control.
        """
        return "TEACHING" in self.control_mode()

    def state(self):
        return dict(t=time.time(), q=self.q(), dq=self.dq(),
                    effort=self.effort(), current=self.current())

    # ---------- control ----------

    def clamp(self, q):
        return np.clip(q, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])

    def move_j(self, q, speed_pct=20):
        q = self.clamp(np.asarray(q, float))
        self.piper.MotionCtrl_2(0x01, 0x01, int(speed_pct), 0x00)
        self.piper.JointCtrl(*[int(round(v * RAD2CMD)) for v in q])

    def hold(self, speed_pct=20):
        self.move_j(self.q(), speed_pct)

    def mit_mode(self):
        """Put the arm in MIT mode. Must be re-sent periodically.

        MIT is engaged through the is_mit_mode field (0xAD) with move_mode left
        at MOVE J, exactly as the vendor demo piper_set_mit.py does. Sending
        move_mode=0x04 (MOVE M) instead makes the arm report MOVE_M but ignore
        the per-joint MIT frames entirely.
        """
        self.piper.MotionCtrl_2(0x01, 0x01, 0, 0xAD)

    def mit(self, i, pos, vel, kp, kd, tau):
        """MIT command for joint i (0-indexed)."""
        tau = float(np.clip(tau, -MIT_TORQUE_LIMIT, MIT_TORQUE_LIMIT))
        self.piper.JointMitCtrl(i + 1, float(pos), float(vel), float(kp), float(kd), tau)

    def standby(self):
        """Leave CAN control mode without cutting torque."""
        self.piper.MotionCtrl_2(0x00, 0x00, 0, 0x00)

    def stop(self):
        self.piper.MotionCtrl_1(0x01, 0, 0)   # emergency stop
