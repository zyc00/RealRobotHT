"""Piper transport startup for B601 drag, preserving the configured mount."""
import time

from piperx_teleop import TorqueSession


class PiperDragSession(TorqueSession):
    def _ensure_ready(self):
        guard = getattr(getattr(self, 'law', None), 'guard', None)
        if guard is None:
            return super()._ensure_ready()
        from position_follower import fresh_q
        # In paired mode do not auto-reposition using the SDK's model-only
        # limits: those can be wider than the queried common envelope.
        guard.check(fresh_q(self.arm, time.time()))
        if not all(self.arm.is_enabled()):
            self.piper.EnableArm(7)
            time.sleep(1.5)
        guard.check(fresh_q(self.arm, time.time()))

    def _enter_mit(self, timeout=2.5):
        guard = getattr(getattr(self, 'law', None), 'guard', None)
        if guard is not None:
            from position_follower import fresh_q
            guard.check(fresh_q(self.arm, time.time()))
        # installation_pos=0 means unchanged. Reasserting upright (1) on
        # mode entry can suspend feedback while the firmware reinitializes.
        sent_at = time.time()
        self.piper.MotionCtrl_2(0x01, 0x06, 0, 0xAD, 0, 0x00)
        deadline = time.monotonic() + timeout
        ready_since = None
        while time.monotonic() < deadline:
            status = self.piper.GetArmStatus()
            mode = getattr(status.arm_status.mode_feed, 'value', status.arm_status.mode_feed)
            stamp = self.arm.obs_time()
            now = time.time()
            ready = (int(mode) == 6 and status.time_stamp > sent_at
                     and stamp > sent_at and -0.01 <= now-stamp <= 0.05)
            if ready:
                ready_since = time.monotonic() if ready_since is None else ready_since
                if time.monotonic()-ready_since >= 0.05:
                    return True
            else:
                ready_since = None
            time.sleep(0.01)
        return False
