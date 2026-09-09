"""Opt-in experimental MIT PD follower. No leader payload calibration reused."""
import time

import numpy as np

from position_follower import PositionFollower, fresh_q


class MitFollower(PositionFollower):
    mode_name = 'mit'
    expected_mode = 6

    def __init__(self, *args, kp=10., kd=.8, **kwargs):
        super().__init__(*args, **kwargs)
        self.kp, self.kd = 10., .8
        self.set_gain('follower_kp', kp)
        self.set_gain('follower_kd', kd)
        self.applied_kp, self.applied_kd = self.kp, self.kd
        self.gain_rows = []

    def set_gain(self, name, value):
        ranges = {'follower_kp': (1., 30.), 'follower_kd': (.1, 2.)}
        if name not in ranges or not np.isfinite(value) or not ranges[name][0] <= value <= ranges[name][1]:
            raise ValueError('invalid follower MIT gain')
        with self.settings_lock:
            setattr(self, 'kp' if name == 'follower_kp' else 'kd', float(value))
        return value

    def speed_settings(self):
        with self.settings_lock:
            return dict(follower_speed=self.max_speed, follower_speed_pct=self.speed_pct,
                        follower_kp=self.kp, follower_kd=self.kd)

    def set_speed(self, name, value):
        if name == 'follower_speed_pct':
            raise ValueError('firmware speed percentage does not apply to MIT tracking')
        return super().set_speed(name, value)

    def _mit_send(self, q):
        for j in range(6):
            self.arm.piper.JointMitCtrl(j+1, float(q[j]), 0., self.applied_kp, self.applied_kd, 0.)

    def start(self):
        self._check_follower_pose(fresh_q(self.arm, time.time()))
        # Enable/hold in MOVE_J first. Never switch a disabled arm straight to MIT.
        super().start()
        self.command = fresh_q(self.arm, time.time())
        sent = time.time()
        self.arm.piper.MotionCtrl_2(1, 6, 0, 0xAD, 0, 0)
        deadline = time.monotonic()+2.5
        ready_since = None
        while time.monotonic() < deadline:
            fresh_q(self.leader, time.time(), self.stale_s)
            actual = fresh_q(self.arm, time.time(), self.stale_s)
            self._check_actual(actual)
            self._mit_send(self.command)
            reply = self.arm.piper.GetArmStatus()
            mode = getattr(reply.arm_status.mode_feed, 'value', reply.arm_status.mode_feed)
            now = time.monotonic()
            if int(mode) == 6 and reply.time_stamp > sent and self.arm.obs_time() > sent:
                ready_since = now if ready_since is None else ready_since
                if now-ready_since >= .05:
                    self.previous = now
                    return
            else:
                ready_since = None
            time.sleep(.01)
        raise RuntimeError('follower failed to enter MIT with fresh feedback')

    def _check_actual(self, actual):
        self._check_follower_pose(actual)
        if not all(self.arm.is_enabled()) or getattr(self.arm.status(), 'err_code', 0):
            raise RuntimeError('follower MIT disabled motor or firmware fault')
        if np.max(abs(actual-self.command)) > np.radians(5):
            raise RuntimeError('follower MIT tracking error exceeds 5 deg; stopping')

    def tick(self, now=None, monotonic=None):
        wall = time.time() if now is None else now
        stamp = time.monotonic() if monotonic is None else monotonic
        if not self.active:
            raise RuntimeError('follower has not started')
        if self.previous is None or not 0 <= stamp-self.previous <= .1:
            raise RuntimeError('follower MIT control-loop deadline missed')
        self._check_actual(fresh_q(self.arm, wall, self.stale_s))
        mode = self.arm.status().mode_feed
        if int(getattr(mode, 'value', mode)) != 6:
            raise RuntimeError('follower lost MIT mode')
        super().tick(now=wall, monotonic=stamp)

    def _send_tracking(self, q, actual, dt, settings):
        self._check_actual(actual)
        # Slew gain changes as well as position targets; never emit HTTP-thread CAN.
        self.applied_kp += float(np.clip(settings['follower_kp']-self.applied_kp, -5*dt, 5*dt))
        self.applied_kd += float(np.clip(settings['follower_kd']-self.applied_kd, -.5*dt, .5*dt))
        self._mit_send(q)
        if self.log_enabled:
            self.gain_rows.append([self.applied_kp, self.applied_kd])

    def save_log(self, path):
        np.savez(path, data=np.asarray(self.rows), mode='mit',
                 columns=np.array('wall_time leader_q(6) follower_q(6) command_q(6)'.split()),
                 speed_settings=np.asarray(self.speed_rows),
                 speed_columns=np.array('target_rad_s unused_movej_percent'.split()),
                 applied_gains=np.asarray(self.gain_rows), gain_columns=['kp', 'kd'],
                 joint_offset_rad=self.offset,
                 leader_can=self.leader.can_name, follower_can=self.arm.can_name)

    def stop(self):
        # Inherited stop explicitly switches to MOVE_J and sends position hold,
        # not MIT commands. No release/disable and no leader tool model used.
        super().stop()
