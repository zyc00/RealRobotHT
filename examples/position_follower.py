"""Six-joint leader mirroring through Piper MOVE_J position control only.

No follower gravity model, feed-forward torque, MIT command or gripper command.
"""
import time
import threading

import numpy as np

from piperx_teleop.arm import JOINT_LIMITS, RAD2CMD


def fresh_q(arm, now, max_age=0.15):
    stamp = arm.obs_time()
    q = np.asarray(arm.q(), float)
    if not np.isfinite(stamp) or stamp <= 0 or not -0.05 <= now-stamp <= max_age:
        raise RuntimeError('stale joint feedback on %s (age %.3f s, limit %.3f s)' %
                           (arm.can_name, now-stamp, max_age))
    if q.shape != (6,) or not np.isfinite(q).all():
        raise RuntimeError('invalid joint feedback on ' + arm.can_name)
    return q


def prepare_feedback(arm, timeout=2.0):
    """Recover a silent follower's normal output stream, without motor commands.

    Separate buses use default feedback/control IDs. Query replies alone do
    not guarantee periodic joint feedback. Do not reset a healthy connection.
    """
    try:
        return fresh_q(arm, time.time())
    except RuntimeError:
        pass
    print('Restoring normal follower feedback on', arm.can_name)
    arm.piper.MasterSlaveConfig(0xFC, 0x00, 0x00, 0x00)
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        try:
            return fresh_q(arm, time.time())
        except RuntimeError:
            time.sleep(0.02)
    raise RuntimeError('no fresh joint feedback on %s after restoring output mode; '
                       'check arm power, CAN connection and teach mode' % arm.can_name)


def checked_target(q):
    q = np.asarray(q, float)
    if q.shape != (6,) or not np.isfinite(q).all():
        raise RuntimeError('invalid leader target')
    # Permit encoder rounding at the limits, not unreachable leader targets.
    if np.any(q < JOINT_LIMITS[:, 0]-0.002) or np.any(q > JOINT_LIMITS[:, 1]+0.002):
        raise RuntimeError('leader pose is outside follower joint limits')
    return np.clip(q, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])


class PositionFollower:
    mode_name = 'position'
    expected_mode = 1
    def __init__(self, leader, follower, max_speed=0.5, speed_pct=20, stale_s=0.15,
                 max_error=np.radians(20), log=False, guard=None, offset=None, follower_guard=None):
        if leader.can_name == follower.can_name:
            raise ValueError('leader and follower must use different CAN interfaces')
        if not 0 < max_speed <= 2 or not 1 <= speed_pct <= 100 or not 0 < stale_s <= 1:
            raise ValueError('invalid follower speed or feedback timeout')
        self.leader, self.arm = leader, follower
        self.max_speed, self.speed_pct, self.stale_s = max_speed, speed_pct, stale_s
        self.max_error = max_error
        self.command = None
        self.active = False
        self.error_since = None
        self.mode_since = None
        self.previous = None
        self.rows = []
        self.log_enabled = log
        self.telem = {}
        self.guard = guard
        self.offset = np.zeros(6) if offset is None else np.asarray(offset, float).copy()
        if self.offset.shape != (6,) or not np.isfinite(self.offset).all():
            raise ValueError('invalid follower joint offset')
        self.follower_guard = follower_guard if follower_guard is not None else guard
        self.settings_lock = threading.Lock()
        self.applied_speed_pct = None
        self.speed_rows = []

    def speed_settings(self):
        with self.settings_lock:
            return dict(follower_speed=self.max_speed, follower_speed_pct=self.speed_pct)

    def set_speed(self, name, value):
        if not np.isfinite(value):
            raise ValueError('invalid follower speed')
        with self.settings_lock:
            if name == 'follower_speed' and 0 < value <= 2:
                self.max_speed = float(value)
            elif name == 'follower_speed_pct' and 1 <= value <= 100 and value == int(value):
                self.speed_pct = int(value)
            else:
                raise ValueError('invalid follower speed parameter or range')
        # No CAN writes from the HTTP thread. The control loop applies this.
        return value

    def _target(self, q):
        if self.guard is not None:
            self.guard.check(q)
        target = np.asarray(q, float) + self.offset
        self._check_follower_pose(target)
        return target.copy()

    def _check_follower_pose(self, q):
        if self.follower_guard is not None:
            self.follower_guard.check(q, 'follower')
        else:
            checked_target(q)

    def _position_mode(self, speed_pct=None):
        # Preserve the firmware's installed mounting configuration. PiperArm
        # move_j() asserts installation_pos=1 on its first call; on the follower
        # firmware that pauses joint feedback for >200 ms, tripping the 150 ms
        # watchdog. Position following needs no gravity/mount reconfiguration.
        if speed_pct is None:
            speed_pct = self.speed_settings()['follower_speed_pct']
        self.arm.piper.MotionCtrl_2(0x01, 0x01, speed_pct, 0x00, 0, 0x00)
        self.applied_speed_pct = speed_pct

    def _send_position(self, q):
        # The desired leader pose was already checked against joint limits.
        # Preserve measured startup/stop poses (which can sit just beyond the
        # model limits) instead of silently jumping them through arm.clamp().
        self.arm.piper.JointCtrl(*[int(round(v * RAD2CMD)) for v in q])

    def _send_tracking(self, q, actual, dt, settings):
        if settings['follower_speed_pct'] != self.applied_speed_pct:
            self._position_mode(settings['follower_speed_pct'])
        self._send_position(q)

    def start(self):
        now = time.time()
        self._target(fresh_q(self.leader, now, self.stale_s))
        self.command = fresh_q(self.arm, now, self.stale_s)
        if self.arm.in_teach_mode():
            raise RuntimeError('follower is in teach mode; exit it before starting')
        # Select normal MOVE_J and seed current pose before enabling; never MIT.
        self.active = True
        self._position_mode()
        self._send_position(self.command)
        deadline = time.monotonic()+5
        while not all(self.arm.is_enabled()):
            if time.monotonic() > deadline:
                raise RuntimeError('follower motors did not enable')
            fresh_q(self.leader, time.time(), self.stale_s)
            fresh_q(self.arm, time.time(), self.stale_s)
            self.arm.piper.EnableArm(7)
            self._send_position(self.command)
            time.sleep(0.02)
        self.previous = time.monotonic()

    def tick(self, now=None, monotonic=None):
        now = time.time() if now is None else now
        monotonic = time.monotonic() if monotonic is None else monotonic
        if not self.active:
            raise RuntimeError('follower has not started')
        leader_q = fresh_q(self.leader, now, self.stale_s)
        target = self._target(leader_q)
        actual = fresh_q(self.arm, now, self.stale_s)
        if not all(self.arm.is_enabled()):
            raise RuntimeError('follower motor disabled while mirroring')
        status = self.arm.status()
        mode = getattr(status.mode_feed, 'value', status.mode_feed)
        # MOVE_J is mode_feed=1. A grace period covers feedback after entry.
        if int(mode) != self.expected_mode:
            self.mode_since = monotonic if self.mode_since is None else self.mode_since
            if monotonic-self.mode_since > 0.5:
                raise RuntimeError('follower did not stay in %s mode' % self.mode_name)
        else:
            self.mode_since = None
        dt = min(max(monotonic-self.previous, 0), 0.05)
        self.previous = monotonic
        if np.max(np.abs(actual-self.command)) > self.max_error:
            self.error_since = monotonic if self.error_since is None else self.error_since
            if monotonic-self.error_since > 0.5:
                raise RuntimeError('follower tracking error persisted; check obstruction or control mode')
        else:
            self.error_since = None
        settings = self.speed_settings()
        max_speed, speed_pct = settings['follower_speed'], settings['follower_speed_pct']
        self.command = self.command + np.clip(target-self.command, -max_speed*dt, max_speed*dt)
        self._send_tracking(self.command, actual, dt, settings)
        self.telem = dict(can=self.arm.can_name, mode=self.mode_name, leader=leader_q.tolist(),
                          mapped_target=target.tolist(),
                          joint_offset_deg=np.degrees(self.offset).tolist(),
                          actual=actual.tolist(), target=self.command.tolist(),
                          max_error_deg=float(np.degrees(np.max(abs(actual-target)))))
        if self.log_enabled:
            self.rows.append(np.concatenate([[now], leader_q, actual, self.command]))
            self.speed_rows.append([max_speed, speed_pct])

    def stop(self):
        if not self.active:
            return
        self.active = False
        # Prefer a fresh measured pose; on loss of feedback, freeze the last
        # commanded target rather than issuing an unbounded/new destination.
        try:
            q = fresh_q(self.arm, time.time(), self.stale_s)
        except RuntimeError:
            q = self.command
        if q is not None:
            self._position_mode()
            for _ in range(3):
                self._send_position(q)
                time.sleep(0.01)

    def save_log(self, path):
        np.savez(path, data=np.asarray(self.rows), columns=np.array('wall_time leader_q(6) follower_q(6) command_q(6)'.split()),
                 speed_settings=np.asarray(self.speed_rows), speed_columns=np.array('target_rad_s firmware_percent'.split()),
                 joint_offset_rad=self.offset,
                 leader_can=self.leader.can_name, follower_can=self.arm.can_name, mode='position')
