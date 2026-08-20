"""Cartesian target sink for the Piper, driven by incremental deltas.

Uses the FIRMWARE's own end-pose feedback as the reference, so we never have to
resolve the ~0.115 m disagreement between our URDF FK and the firmware's end
pose - every target is "where the firmware says we are, plus a small delta".

The firmware runs its own IK, which we cannot inspect, so this wraps it with:

  * per-cycle step clamp        - a teleop delta can never become a lunge
  * workspace box + z floor     - relative to where the session started
  * joint-space watchdog        - a small Cartesian step MUST produce a small
                                  joint step; a big one means the firmware
                                  picked a different IK branch or hit a
                                  singularity, and we abort to joint hold
  * low speed by default        - the vendor demos use 100%, which is reckless

Rotation is locked by default: the SDK documents RX/RY/RZ as Euler angles but
not which convention, and guessing it would silently corrupt orientation.

Measured on this arm (scripts/13_test_endpose.py, 50 mm legs at 100 mm/s):

  * tracking in well-conditioned regions is excellent - 100-101% of the
    commanded motion on +/-X and +/-Z
  * per-cycle joint motion is smooth: median 0.40 deg, p99 0.81 deg, max 0.81
  * but IK pathologies are real and reproducible.  Two separate runs aborted:
    J3 by 5.9 deg in one cycle moving -Z, and J6 by 17.9 deg moving -Y, the
    latter 22x the p99 and accompanied by REACH_TARGET_POS_FAILED and a visible
    wrist reconfiguration.  Lateral (+/-Y) motion degraded to 85% tracking just
    before failing.

So the firmware is ACCURATE but NOT ROBUST: it will happily reconfigure through
a branch flip mid-motion.  The watchdog is not optional, and 2 deg (~2.5x the
measured p99) catches a flip long before it becomes a swing.
"""

import time

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from .arm import JOINT_LIMITS

MOVE_P = 0x00
MOVE_L = 0x02

# Identified on this arm by scripts/16_euler_selfconsistent.py: sweeping a single
# wrist joint must produce a rotation about a constant axis through the joint's
# own angle.  Extrinsic "xyz" with angles (RX, RY, RZ) - i.e. fixed-axis
# roll/pitch/yaw - reproduced that with 0.00 deg angle error and 0.00 deg axis
# spread; the next best convention was 16 deg off.
EULER_SEQ = "xyz"
# Extrinsic xyz is singular at RY = +/-90 deg, where RX and RZ stop being
# separable and small orientation changes produce huge angle changes.
GIMBAL_WARN_DEG = 80.0


class CartesianController:
    def __init__(self, arm,
                 max_step_m=0.004,          # per control cycle
                 max_reach_m=0.30,          # box half-size around the session origin
                 min_z_m=0.05,              # absolute floor, firmware frame
                 max_joint_step_deg=2.0,    # watchdog; see the note below
                 max_rot_step_deg=1.5,      # per-cycle orientation change
                 lock_rotation=True,
                 speed_pct=20,
                 move_mode=MOVE_P,
                 position_gain=1.0,
                 lead_limit_m=0.04,
                 dry_run=False):
        self.arm = arm
        self.max_step = max_step_m
        self.max_reach = max_reach_m
        self.min_z = min_z_m
        self.max_joint_step = np.radians(max_joint_step_deg)
        self.max_rot_step = np.radians(max_rot_step_deg)
        self.lock_rotation = lock_rotation
        self.clamp_rot = 0
        self.gimbal_warned = False
        self.speed_pct = speed_pct
        self.move_mode = move_mode
        self.gain = position_gain
        # How far the command may lead the arm's ACTUAL pose.  Without this, a
        # stall against a joint limit lets the command run away, and the
        # operator then has to move all the way back through the gap before
        # anything responds - which feels like total unresponsiveness.
        self.lead_limit = lead_limit_m
        self.dry_run = dry_run

        self.origin = None        # firmware pose at session start
        self.target = None        # [x,y,z] metres, firmware frame
        self.rot = None           # [rx,ry,rz] degrees
        self.anchor_pos = None    # arm pose when the clutch last engaged
        self.anchor_rot = None
        self.cmd_pos = None       # rate-limited command chasing the goal
        self.cmd_rot = None
        self._last_q = None
        self.aborted = None
        self.clamp_events = 0
        # Broken out, because they mean different things: step = you moved
        # faster than the rate limit (arm lags), box/floor = you hit the edge
        # of the allowed region (arm stops).
        self.clamp_step = 0
        self.clamp_box = 0
        self.clamp_floor = 0
        self._announced = set()
        self.notes = []

    # ---------- firmware pose ----------

    def read_pose(self):
        ep = self.arm.piper.GetArmEndPoseMsgs().end_pose
        pos = np.array([ep.X_axis, ep.Y_axis, ep.Z_axis], float) * 1e-6      # 0.001mm -> m
        rot = np.array([ep.RX_axis, ep.RY_axis, ep.RZ_axis], float) * 1e-3   # 0.001deg -> deg
        return pos, rot

    def start(self):
        pos, rot = self.read_pose()
        self.origin = pos.copy()
        self.target = pos.copy()
        self.rot = rot.copy()
        self.anchor_pos = pos.copy()
        self.anchor_rot = rot.copy()
        self.cmd_pos = pos.copy()
        self.cmd_rot = rot.copy()
        self._last_q = self.arm.q()
        self.aborted = None
        return self

    def relatch(self):
        """Anchor to where the arm actually is (call on every clutch engage)."""
        pos, rot = self.read_pose()
        self.target = pos.copy()
        self.rot = rot.copy()
        self.anchor_pos = pos.copy()
        self.anchor_rot = rot.copy()
        self.cmd_pos = pos.copy()
        self.cmd_rot = rot.copy()
        self._last_q = self.arm.q()

    def follow(self, disp_pos, disp_rotvec=None):
        """Anchored absolute mapping - the correct way to drive this.

        `disp_pos` is the operator's TOTAL displacement since the clutch engaged,
        so the goal pose is recomputed from the anchor every tick rather than
        accumulated.  The rate limit then applies to how fast the COMMAND chases
        that goal, instead of permanently deleting motion the way clamping an
        integrated target does.
        """
        if self.aborted:
            return None
        goal = self.anchor_pos + np.asarray(disp_pos, float) * self.gain
        goal = np.clip(goal, self.origin - self.max_reach, self.origin + self.max_reach)
        if goal[2] < self.min_z:
            goal[2] = self.min_z
            self._announce("floor", "Z FLOOR reached (z=%.3f m); cannot go lower." % self.min_z)
        if np.any(np.abs(goal - self.origin) >= self.max_reach - 1e-9):
            self._announce("box", "WORKSPACE BOX limit (+/-%.2f m from session start). "
                                  "Release the clutch, move your hand back, re-clutch."
                           % self.max_reach)

        step = goal - self.cmd_pos
        n = np.linalg.norm(step)
        if n > self.max_step:
            step = step / n * self.max_step
            self.clamp_step += 1
            self.clamp_events += 1
        self.cmd_pos = self.cmd_pos + step

        # Leash the command to the arm's real position.
        actual, _ = self.read_pose()
        lead = self.cmd_pos - actual
        ln = np.linalg.norm(lead)
        if ln > self.lead_limit:
            self.cmd_pos = actual + lead / ln * self.lead_limit
            self._announce("stall",
                           "ARM STALLED: it is not reaching the commanded pose (held "
                           "%.0f mm behind). Usually a joint limit - check the JOINT "
                           "LIMIT note, or move back the way you came."
                           % (self.lead_limit * 1000))
        self.target = self.cmd_pos

        if disp_rotvec is not None and not self.lock_rotation:
            R_goal = Rot.from_rotvec(np.asarray(disp_rotvec, float)) * \
                     Rot.from_euler(EULER_SEQ, self.anchor_rot, degrees=True)
            R_cmd = Rot.from_euler(EULER_SEQ, self.cmd_rot, degrees=True)
            err = (R_goal * R_cmd.inv()).as_rotvec()
            m = np.linalg.norm(err)
            if m > self.max_rot_step:
                err = err / m * self.max_rot_step
                self.clamp_rot += 1
            self.cmd_rot = (Rot.from_rotvec(err) * R_cmd).as_euler(EULER_SEQ, degrees=True)
            self.rot = self.cmd_rot
            if abs(self.rot[1]) > GIMBAL_WARN_DEG and not self.gimbal_warned:
                self.gimbal_warned = True
                print("  !! near gimbal lock (RY=%.0f deg)" % self.rot[1])

        self._send()
        return self.target

    def _send(self):
        if self.dry_run:
            return
        self.arm.piper.MotionCtrl_2(0x01, self.move_mode, int(self.speed_pct), 0x00)
        self.arm.piper.EndPoseCtrl(
            int(round(self.target[0] * 1e6)), int(round(self.target[1] * 1e6)),
            int(round(self.target[2] * 1e6)), int(round(self.rot[0] * 1e3)),
            int(round(self.rot[1] * 1e3)), int(round(self.rot[2] * 1e3)))

    # ---------- safety ----------

    def _clamp_target(self, t):
        step = t - self.target
        n = np.linalg.norm(step)
        if n > self.max_step:
            step = step / n * self.max_step
            self.clamp_events += 1
            self.clamp_step += 1
            self._announce("step",
                           "STEP LIMIT active (%.0f mm/cycle): the arm is following but "
                           "lagging your hand. Lower --gain or raise --max-step."
                           % (self.max_step * 1000))
        t = self.target + step
        off = t - self.origin
        big = np.abs(off) > self.max_reach
        if big.any():
            axes = "".join("xyz"[i] for i in range(3) if big[i])
            off = np.clip(off, -self.max_reach, self.max_reach)
            t = self.origin + off
            self.clamp_events += 1
            self.clamp_box += 1
            self._announce("box_" + axes,
                           "WORKSPACE BOX limit on %s (+/-%.2f m from where the session "
                           "started). Release the clutch, reposition, re-clutch - or raise "
                           "--max-reach." % (axes, self.max_reach))
        if t[2] < self.min_z:
            t[2] = self.min_z
            self.clamp_events += 1
            self.clamp_floor += 1
            self._announce("floor",
                           "Z FLOOR reached (z=%.3f m). The arm cannot go lower; "
                           "raise it with --min-z or start higher (--home)." % self.min_z)
        return t

    def _announce(self, key, msg):
        if key not in self._announced:
            self._announced.add(key)
            self.notes.append(msg)

    def tracking_error(self):
        """How far the arm actually is from the commanded target (metres)."""
        pos, _ = self.read_pose()
        return float(np.linalg.norm(self.target - pos)), pos

    def check_joints(self):
        """True if the arm is tracking sanely; sets self.aborted otherwise.

        Also warns when a joint runs out of travel.  The firmware's IK will
        happily straighten the elbow to satisfy a descending target and drive J3
        into its 0 deg limit - measured at 336 mm of descent from the standard
        home pose - after which the arm simply stops following with no error of
        its own.  Naming the joint makes that legible instead of mysterious.
        """
        q = self.arm.q()
        margin = np.degrees(np.minimum(q - JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1] - q))
        near = np.where(margin < 3.0)[0]
        for j in near:
            self._announce("limit%d" % j,
                           "JOINT LIMIT: J%d is %.1f deg from its end of travel "
                           "(%.1f deg). The firmware's IK has run out of room this way. "
                           "Re-clutching will NOT help - the arm is still against the "
                           "stop. Move back the way you came, or restart with --home."
                           % (j + 1, margin[j], np.degrees(q[j])))
        if self._last_q is not None:
            jump = np.abs(q - self._last_q)
            if jump.max() > self.max_joint_step:
                self.aborted = ("joint %d jumped %.1f deg in one cycle - likely an IK "
                                "branch flip or singularity"
                                % (int(jump.argmax()) + 1, np.degrees(jump.max())))
                return False
        self._last_q = q
        return True

    def hold_joints(self):
        """Escape hatch: leave Cartesian mode, pin the arm in joint space."""
        q = self.arm.q()
        for _ in range(25):
            self.arm.move_j(q, speed_pct=10)
            time.sleep(0.01)

    # ---------- command ----------

    def _apply_rotation(self, drot):
        """Compose a base-frame rotation delta onto the held orientation."""
        rv = np.asarray(drot, float)
        n = np.linalg.norm(rv)
        if n < 1e-9:
            return
        if n > self.max_rot_step:
            rv = rv / n * self.max_rot_step
            self.clamp_rot += 1
        R_cur = Rot.from_euler(EULER_SEQ, self.rot, degrees=True)
        R_new = Rot.from_rotvec(rv) * R_cur
        self.rot = R_new.as_euler(EULER_SEQ, degrees=True)
        if abs(self.rot[1]) > GIMBAL_WARN_DEG and not self.gimbal_warned:
            self.gimbal_warned = True
            print("  !! near gimbal lock (RY=%.0f deg): orientation control is "
                  "ill-conditioned here" % self.rot[1])

    def apply_delta(self, dpos, drot=None):
        """Advance the target by a robot-base delta and command it."""
        if self.aborted:
            return None
        if drot is not None and not self.lock_rotation:
            self._apply_rotation(drot)
        want = self.target + np.asarray(dpos, float) * self.gain
        self.target = self._clamp_target(want)
        if not self.dry_run:
            self.arm.piper.MotionCtrl_2(0x01, self.move_mode, int(self.speed_pct), 0x00)
            self.arm.piper.EndPoseCtrl(
                int(round(self.target[0] * 1e6)),
                int(round(self.target[1] * 1e6)),
                int(round(self.target[2] * 1e6)),
                int(round(self.rot[0] * 1e3)),
                int(round(self.rot[1] * 1e3)),
                int(round(self.rot[2] * 1e3)),
            )
        return self.target

    def set_gripper(self, closed, effort=1000, opening_m=0.06):
        if self.dry_run:
            return
        angle = 0 if closed else int(opening_m * 1e6)
        self.arm.piper.GripperCtrl(abs(angle), effort, 0x01, 0)
