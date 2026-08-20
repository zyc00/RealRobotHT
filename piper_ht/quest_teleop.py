"""Quest 3 right-controller -> Cartesian delta commands.

Depends only on `robovr.quest3` for raw headset/controller state; the teleop
mapping (clutch semantics, frames, deadbands) lives here.

Coordinate convention, matching the RoboVR examples:

    Quest/OpenXR world:  +X right, +Y up, -Z forward   (gravity-aligned)
    Piper base:          +X forward, +Y left, +Z up

so  x_robot = +forward,  y_robot = -right,  z_robot = +up.

The Quest world frame's YAW is arbitrary - it is fixed when tracking starts -
so `yaw_offset_deg` rotates the horizontal axes to line the operator up with the
robot. With the headset resting on a table (not worn), leave `head_relative`
False: the neutral frame then stays the fixed, gravity-aligned world frame,
which is exactly what you want. Deriving it from a headset lying face-down on a
desk would put "forward" into the floor.

Motion is incremental and clutched, so the frame ORIGIN never matters - only the
axes do.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

POSITION_VALID_BIT = 0x2


@dataclass
class QuestTeleopConfig:
    host: str = "127.0.0.1"
    port: int = 7777
    adb_reverse: bool = True
    adb_path: str = "adb"
    timeout_s: float = 1.0

    clutch_threshold: float = 0.5      # right squeeze
    gripper_threshold: float = 0.75    # right trigger (toggles)
    position_deadband_m: float = 0.001
    rotation_deadband_rad: float = 0.01
    # Deadband on the ANCHORED rotation, in degrees.  Holding a controller and
    # moving your hand rotates it incidentally by tens of degrees; without this
    # every twitch drives the wrist.  Measured in a real session: J4 swung 122
    # deg and the rotation rate limit saturated on 39% of ticks.
    rot_disp_deadband_deg: float = 6.0

    yaw_offset_deg: float = 0.0        # align Quest world yaw with the robot base
    forward_sign: float = 1.0
    right_sign: float = 1.0
    up_sign: float = 1.0
    head_yaw_align: bool = True        # take the frame heading from the headset's yaw
    head_relative: bool = False        # full headset orientation (wrong for a headset on a table)
    relatch_on_button_a: bool = True
    # How the frame heading follows the headset:
    #   "between_clutch" - re-aim whenever the clutch is OPEN, hold it while you
    #                      are dragging.  Correct for a WORN headset: "forward"
    #                      is always where you are looking when you start a
    #                      motion, but the mapping cannot rotate underneath you
    #                      mid-drag.  Also correct for a static headset, which
    #                      simply never changes.
    #   "continuous"     - re-aim every tick, including mid-drag.  Turning your
    #                      head while clutched curves the robot's path.
    #   "latched"        - aim once (and on A).  Only right for a headset that
    #                      never moves.
    frame_mode: str = "between_clutch"
    lock_rotation: bool = True         # position-only until the firmware Euler convention is pinned down


@dataclass
class TeleopSample:
    connected: bool = False
    clutch: bool = False
    reset_reference: bool = False
    dpos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    drot: np.ndarray = field(default_factory=lambda: np.zeros(3))
    dpos_world: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Total displacement from the pose held when the clutch engaged (robot base
    # frame).  This is the quantity to drive the arm with - see below.
    disp_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    disp_rotvec: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gripper_closed: bool = False
    tracking_valid: bool = False
    heading_deg: float = 0.0


def _norm_quat(q):
    q = np.asarray(q, float)
    n = np.linalg.norm(q)
    return np.array([0.0, 0.0, 0.0, 1.0]) if n < 1e-12 else q / n


def _quat_conj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def _quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def _quat_to_rotvec(q):
    q = _norm_quat(q)
    if q[3] < 0:
        q = -q
    v = q[:3]
    s = np.linalg.norm(v)
    if s < 1e-9:
        return np.zeros(3)
    return v / s * (2.0 * np.arctan2(s, q[3]))


def _rotate_by_quat(v, q):
    qv = np.array([v[0], v[1], v[2], 0.0])
    return _quat_mul(_quat_mul(q, qv), _quat_conj(q))[:3]


class QuestNeutralFrame:
    """Axes the controller deltas are projected onto."""

    def __init__(self, yaw_offset_deg=0.0):
        self._right = np.array([1.0, 0.0, 0.0])
        self._up = np.array([0.0, 1.0, 0.0])
        self._forward = np.array([0.0, 0.0, -1.0])
        self.set_yaw_offset(yaw_offset_deg)

    def set_yaw_offset(self, deg):
        """Rotate the horizontal axes about world up (+Y) by `deg`."""
        a = np.radians(float(deg))
        c, s = np.cos(a), np.sin(a)
        self._right = np.array([c, 0.0, -s])
        self._up = np.array([0.0, 1.0, 0.0])
        self._forward = np.array([-s, 0.0, -c])

    def set_from_head_yaw(self, quat_xyzw, extra_yaw_deg=0.0):
        """Align the horizontal axes with where the headset is FACING.

        Uses heading only, so the frame stays gravity-aligned no matter how the
        headset is propped up.  If the headset is lying face-down its forward
        axis points at the floor and has no usable heading - in that case its
        own +Y axis is the one lying horizontally, pointing where the wearer
        would face, so we use that instead.
        """
        q = _norm_quat(quat_xyzw)
        up = np.array([0.0, 1.0, 0.0])
        fwd = _rotate_by_quat(np.array([0.0, 0.0, -1.0]), q)
        horiz = fwd - np.dot(fwd, up) * up
        if np.linalg.norm(horiz) < 0.10:                    # pitched near-vertical (>84 deg)
            hup = _rotate_by_quat(np.array([0.0, 1.0, 0.0]), q)
            horiz = hup - np.dot(hup, up) * up
        n = np.linalg.norm(horiz)
        if n < 1e-6:
            return False
        horiz = horiz / n
        a = np.radians(float(extra_yaw_deg))
        c, sn = np.cos(a), np.sin(a)
        horiz = np.array([horiz[0] * c + horiz[2] * sn, 0.0,
                          -horiz[0] * sn + horiz[2] * c])
        self._forward = horiz
        self._up = up
        self._right = np.cross(horiz, up)
        return True

    def heading_deg(self):
        """Compass-style heading of the frame's forward axis, for reporting."""
        return float(np.degrees(np.arctan2(-self._forward[0], -self._forward[2])))

    def set_from_quat(self, quat_xyzw):
        q = _norm_quat(quat_xyzw)
        self._right = _rotate_by_quat(np.array([1.0, 0.0, 0.0]), q)
        self._up = _rotate_by_quat(np.array([0.0, 1.0, 0.0]), q)
        self._forward = _rotate_by_quat(np.array([0.0, 0.0, -1.0]), q)

    def axes(self):
        return self._right.copy(), self._up.copy(), self._forward.copy()


class QuestCartesianSource:
    """Turns Quest right-controller motion into robot-base Cartesian deltas."""

    def __init__(self, config: Optional[QuestTeleopConfig] = None, server: Any = None):
        self.cfg = config or QuestTeleopConfig()
        self._server = server
        self._owns = server is None
        self.frame = QuestNeutralFrame(self.cfg.yaw_offset_deg)
        self._prev_pos = None
        self._prev_quat = None
        self._anchor_pos = None
        self._anchor_quat = None
        self._was_clutched = False
        self._trigger_was = False
        self._button_a_was = False
        self._yaw_latched = False
        self.gripper_closed = False

    def start(self):
        if self._server is None:
            from robovr.quest3 import Quest3Server
            self._server = Quest3Server(host=self.cfg.host, port=self.cfg.port,
                                        adb_reverse=self.cfg.adb_reverse,
                                        adb_path=self.cfg.adb_path,
                                        timeout_s=self.cfg.timeout_s)
        start = getattr(self._server, "start", None)
        if callable(start):
            start()
        return self

    def stop(self):
        if self._owns and self._server is not None:
            close = getattr(self._server, "close", None)
            if callable(close):
                close()

    def stats(self):
        s = getattr(self._server, "stats", None)
        return s() if callable(s) else {}

    def _reset_reference(self):
        self._prev_pos = None
        self._prev_quat = None
        self._anchor_pos = None
        self._anchor_quat = None
        self._was_clutched = False

    def _to_robot_base(self, v):
        right, up, fwd = self.frame.axes()
        local = np.array([np.dot(v, right), np.dot(v, up), np.dot(v, fwd)])
        return np.array([
            local[2] * self.cfg.forward_sign,
            -local[0] * self.cfg.right_sign,
            local[1] * self.cfg.up_sign,
        ])

    @property
    def yaw_latched(self):
        return self._yaw_latched

    def poll(self) -> TeleopSample:
        st = self._server.latest() if self._server is not None else None
        if st is None or not getattr(st, "connected", False):
            self._reset_reference()
            return TeleopSample(connected=False)

        head = getattr(st, "head", None)
        head_ok = head is not None and bool(getattr(head, "valid", False))
        button_a = bool(getattr(st, "button_a", False))
        a_edge = button_a and not self._button_a_was
        self._button_a_was = button_a

        # gripper: trigger toggles on rising edge
        trig = float(getattr(st, "right_trigger", 0.0)) >= self.cfg.gripper_threshold
        if trig and not self._trigger_was:
            self.gripper_closed = not self.gripper_closed
        self._trigger_was = trig

        grip = getattr(st, "right_grip", None)
        valid = (grip is not None and bool(getattr(grip, "valid", False))
                 and (int(getattr(st, "right_grip_flags", 0)) & POSITION_VALID_BIT) != 0)
        clutch = valid and float(getattr(st, "right_squeeze", 0.0)) >= self.cfg.clutch_threshold
        reset = bool(getattr(st, "button_b", False))

        # Frame update AFTER the clutch is known, so "between_clutch" can hold
        # the heading steady for the duration of a drag.
        if head_ok:
            q_head = np.asarray(head.quat_xyzw, float)
            if self.cfg.head_relative:
                self.frame.set_from_quat(q_head)
                self._yaw_latched = True
            elif self.cfg.head_yaw_align:
                mode = self.cfg.frame_mode
                if mode == "continuous":
                    update = True
                elif mode == "between_clutch":
                    update = (not clutch) or (not self._yaw_latched)
                else:
                    update = (not self._yaw_latched) or (a_edge and self.cfg.relatch_on_button_a)
                if a_edge and self.cfg.relatch_on_button_a:
                    update = True
                if update and self.frame.set_from_head_yaw(q_head, self.cfg.yaw_offset_deg):
                    self._yaw_latched = True

        out = TeleopSample(connected=True, clutch=clutch,
                           gripper_closed=self.gripper_closed, tracking_valid=valid,
                           heading_deg=self.frame.heading_deg())
        if not clutch:
            self._reset_reference()
            return out

        pos = np.asarray(grip.position, float)
        quat = _norm_quat(np.asarray(grip.quat_xyzw, float))
        if self._prev_pos is None or not self._was_clutched or reset:
            self._prev_pos, self._prev_quat = pos, quat
            self._anchor_pos, self._anchor_quat = pos, quat
            self._was_clutched = True
            out.reset_reference = True
            return out

        dp = pos - self._prev_pos
        dq = _quat_mul(quat, _quat_conj(self._prev_quat))
        self._prev_pos, self._prev_quat = pos, quat
        if np.linalg.norm(dp) < self.cfg.position_deadband_m:
            dp = np.zeros(3)
        rv = _quat_to_rotvec(dq)
        if np.linalg.norm(rv) < self.cfg.rotation_deadband_rad:
            rv = np.zeros(3)

        out.dpos_world = dp.copy()
        out.dpos = self._to_robot_base(dp)
        out.drot = np.zeros(3) if self.cfg.lock_rotation else self._to_robot_base(rv)

        # Anchored absolute displacement: the controller's TOTAL motion since the
        # clutch engaged, not a sum of per-tick deltas.  Integrating deltas loses
        # every millimetre a rate limit clips and never recovers it, so the hand
        # and the arm slowly decorrelate; measuring from the anchor makes the
        # mapping self-correcting and immune to dropped samples.
        out.disp_pos = self._to_robot_base(pos - self._anchor_pos)
        if not self.cfg.lock_rotation:
            dq_total = _quat_mul(quat, _quat_conj(self._anchor_quat))
            rvt = _quat_to_rotvec(dq_total)
            mag = np.linalg.norm(rvt)
            db = np.radians(self.cfg.rot_disp_deadband_deg)
            if mag <= db:
                rvt = np.zeros(3)
            else:
                # shrink rather than hard-threshold, so crossing the deadband
                # does not produce a jump
                rvt = rvt / mag * (mag - db)
            out.disp_rotvec = self._to_robot_base(rvt)
        return out
