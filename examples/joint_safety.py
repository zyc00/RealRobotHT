"""Conservative software joint envelope; not a hardware stop/collision monitor."""
import time

import numpy as np

from piperx_teleop.arm import JOINT_LIMITS


def validate_limits(limits):
    limits = np.asarray(limits, float)
    if limits.shape != (6, 2) or not np.isfinite(limits).all() or np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError('invalid six-joint limits')
    return limits.copy()


def read_limits(arm, timeout=1.0):
    """Read all six firmware limits, requiring a fresh matching reply each time."""
    limits = []
    for j in range(1, 7):
        sent = time.time()
        arm.piper.SearchMotorMaxAngleSpdAccLimit(j, 1)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            reply = arm.piper.GetCurrentMotorAngleLimitMaxVel()
            m = reply.current_motor_angle_limit_max_vel
            if reply.time_stamp >= sent and m.motor_num == j:
                limits.append([m.min_angle_limit * 0.1, m.max_angle_limit * 0.1])
                break
            time.sleep(0.005)
        else:
            raise RuntimeError('no fresh joint %d limit reply on %s; refusing to start' % (j, arm.can_name))
    return validate_limits(np.radians(limits))


def common_limits(*limits):
    arrays = np.asarray([validate_limits(JOINT_LIMITS)] + [validate_limits(x) for x in limits])
    return validate_limits(np.column_stack((arrays[:, :, 0].max(axis=0), arrays[:, :, 1].min(axis=0))))


def format_limits(leader, follower, guarded, leader_can, follower_can):
    """Keep queried firmware limits distinct from model and software limits."""
    arrays = [np.degrees(validate_limits(x)) for x in (leader, follower, JOINT_LIMITS, guarded)]
    lines = ['Joint ranges in degrees [min, max]:',
             'Joint  Leader firmware (%s)  Follower firmware (%s)  Model limits          Shared guarded' %
             (leader_can, follower_can)]
    for j in range(6):
        cells = ['[%7.2f, %7.2f]' % tuple(a[j]) for a in arrays]
        lines.append('J%d     ' % (j+1) + '    '.join(cells))
    lines.append('Shared guarded: leader-coordinate envelope after follower offset mapping, model limits and 1 deg margin.')
    return '\n'.join(lines)


class JointGuard:
    def __init__(self, limits, margin=np.radians(1), zone=np.radians(8)):
        self.limits = validate_limits(limits)
        if not np.isfinite([margin, zone]).all() or margin <= 0 or zone <= 0:
            raise ValueError('invalid joint guard margin/zone')
        self.hard = validate_limits(self.limits + [margin, -margin])
        if np.any(np.diff(self.hard, axis=1)[:, 0] <= 2*zone):
            raise ValueError('joint envelope too narrow for safety zones')
        self.zone = zone
        self.cap = np.array([1.0, 1.5, 1.0, 0.5, 0.4, 0.3])
        self.damping = np.array([2., 3., 2., 0.7, 0.6, 0.5])

    def check(self, q, label='leader'):
        q = np.asarray(q, float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise RuntimeError(label + ' invalid joint angles')
        bad = np.flatnonzero((q <= self.hard[:, 0]) | (q >= self.hard[:, 1]))
        if len(bad):
            j = int(bad[0])
            raise RuntimeError('%s J%d %.3f deg outside guarded range [%.3f, %.3f] deg; reposition before restarting' %
                               (label, j+1, np.degrees(q[j]), *np.degrees(self.hard[j])))

    def apply(self, q, v, extra):
        self.check(q)
        v, extra = np.asarray(v, float), np.asarray(extra, float)
        if v.shape != (6,) or extra.shape != (6,) or not np.isfinite([v, extra]).all():
            raise RuntimeError('invalid joint guard velocity/torque')
        lower = np.clip(1-(q-self.hard[:, 0])/self.zone, 0, 1)
        upper = np.clip(1-(self.hard[:, 1]-q)/self.zone, 0, 1)
        # Attenuate outward non-gravity torque, preserve inward assistance.
        safe = extra * (1-np.where(extra >= 0, upper, lower))
        wall = (self.cap*(lower-upper) - self.damping *
                (lower*np.minimum(v, 0) + upper*np.maximum(v, 0)))
        return safe + np.clip(wall, -self.cap, self.cap), np.maximum(lower, upper)
