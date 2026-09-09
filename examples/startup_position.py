"""Confirmed, independently targeted paired entry into the shared envelope."""
import time

import numpy as np

from piperx_teleop.arm import RAD2CMD
from position_follower import fresh_q


def entry_target(q, guard):
    q = np.asarray(q, float)
    if q.shape != (6,) or not np.isfinite(q).all():
        raise RuntimeError('invalid startup pose')
    # Allow only small parked encoder excursions, never arbitrary recovery.
    if np.any(q < guard.limits[:, 0]-np.radians(5)) or np.any(q > guard.limits[:, 1]+np.radians(5)):
        raise RuntimeError('arm more than 5 deg outside common limits; manual recovery required')
    # Minimal entry with 2 degrees of clearance from the hard guard. Do not
    # move all the way past the soft wall merely to begin a paired session.
    return np.clip(q, guard.hard[:, 0]+np.radians(2),
                   guard.hard[:, 1]-np.radians(2))


def position_leader(arm, guard, confirm=input):
    return position_arms([arm], guard, confirm=confirm)


def position_arms(arms, guard, confirm=input):
    arms = list(arms)
    if not arms or len({a.can_name for a in arms}) != len(arms):
        raise ValueError('startup requires distinct CAN interfaces')
    guards = list(guard) if isinstance(guard, (list, tuple)) else [guard]*len(arms)
    if len(guards) != len(arms):
        raise ValueError('one startup guard required per arm')
    initial = [fresh_q(a, time.time()) for a in arms]
    targets = [entry_target(q, g) for q, g in zip(initial, guards)]
    if all(np.max(abs(q-t)) < np.radians(0.1) for q, t in zip(initial, targets)):
        for q, g in zip(initial, guards):
            g.check(q)
        return
    for arm, q, target in zip(arms, initial, targets):
        print(arm.can_name, 'startup current (deg):', np.degrees(q).round(2))
        print(arm.can_name, 'startup target  (deg):', np.degrees(target).round(2))
    print('Minimal entry: 2 deg inside the hard guard; soft resistance may be active afterward.')
    print('Arms position concurrently, independently: MOVE_J at 10% firmware speed; target ramps 3 deg/s.')
    print('A parked pose outside nominal limits first targets the nearest valid boundary (at most 5 deg).')
    if confirm('Clear BOTH workspaces. Type MOVE to position the arms (mirroring is OFF): ').strip() != 'MOVE':
        raise RuntimeError('startup positioning cancelled')
    # Validate both before issuing any motion command to either.
    for arm, q in zip(arms, initial):
        if np.max(abs(fresh_q(arm, time.time())-q)) > np.radians(0.5):
            raise RuntimeError(arm.can_name + ' moved since preview; restart to confirm a new target')
        if arm.in_teach_mode():
            raise RuntimeError(arm.can_name + ' is in teach mode')
    commands = [np.clip(q, g.limits[:, 0], g.limits[:, 1]) for q, g in zip(initial, guards)]
    active = []
    def send(arm, q):
        arm.piper.JointCtrl(*[int(round(v*RAD2CMD)) for v in q])
    try:
        for i, arm in enumerate(arms):
            active.append(i)
            arm.piper.MotionCtrl_2(1, 1, 10, 0, 0, 0)
            send(arm, commands[i])
        deadline = time.monotonic()+5
        while not all(all(a.is_enabled()) for a in arms):
            if time.monotonic() > deadline:
                raise RuntimeError('startup motors did not enable')
            for arm, command in zip(arms, commands):
                fresh_q(arm, time.time())
                if not all(arm.is_enabled()):
                    arm.piper.EnableArm(7)
                send(arm, command)
            time.sleep(0.02)
        start = previous = time.monotonic()
        timeout = 10+max(float(np.max(abs(t-c))) for t, c in zip(targets, commands))/np.radians(3)
        settled = None
        while True:
            now = time.monotonic()
            actuals = [fresh_q(a, time.time()) for a in arms]
            for arm, actual, command in zip(arms, actuals, commands):
                if not all(arm.is_enabled()):
                    raise RuntimeError(arm.can_name + ' motor disabled during startup')
                status = arm.status()
                mode = getattr(status.mode_feed, 'value', status.mode_feed)
                if getattr(status, 'err_code', 0) or (now-start > 0.5 and int(mode) != 1):
                    raise RuntimeError(arm.can_name + ' startup fault or loss of MOVE_J mode')
                if np.max(abs(actual-command)) > np.radians(5.5):
                    raise RuntimeError(arm.can_name + ' startup tracking error exceeds 5.5 deg')
            if now-start > timeout:
                raise RuntimeError('startup timed out; mirroring was not started')
            dt = min(max(now-previous, 0), 0.05)
            previous = now
            # Pause the ramp rather than let a stalled arm build target error.
            for arm, actual, command, target in zip(arms, actuals, commands, targets):
                if np.max(abs(actual-command)) < np.radians(1):
                    command += np.clip(target-command, -np.radians(3)*dt, np.radians(3)*dt)
                send(arm, command)
            if all(np.max(abs(q-t)) < np.radians(0.5) for q, t in zip(actuals, targets)):
                settled = now if settled is None else settled
                if now-settled >= 0.3:
                    for arm, actual, g in zip(arms, actuals, guards):
                        g.check(actual, arm.can_name)
                    print('Both startup targets reached; arms holding. Mirroring has not started yet.')
                    return
            else:
                settled = None
            time.sleep(0.02)
    finally:
        for i in active:
            arm, command = arms[i], commands[i]
            guard = guards[i]
            # Best-effort hold, including on Ctrl-C. Never disable/drop the arm.
            try:
                hold = np.clip(fresh_q(arm, time.time()), guard.limits[:, 0], guard.limits[:, 1])
            except Exception:
                hold = command
            try:
                send(arm, hold)
            except Exception as exc:
                # One failed CAN bus must not prevent the other hold attempt.
                print('WARNING: startup hold failed on', arm.can_name, str(exc))
