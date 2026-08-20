"""Offline checks of the Quest -> Cartesian pipeline (no hardware, no headset)."""
import sys
import numpy as np
sys.path.insert(0, ".")
from piper_ht.quest_teleop import QuestCartesianSource, QuestTeleopConfig
from piper_ht.cartesian import CartesianController


class FakePose:
    def __init__(self, p, q=(0, 0, 0, 1), valid=True):
        self.position = np.asarray(p, float)
        self.quat_xyzw = np.asarray(q, float)
        self.valid = valid


class FakeState:
    def __init__(self, p, squeeze=0.0, trigger=0.0, a=False, b=False, head_yaw=0.0):
        self.connected = True
        self.right_grip = FakePose(p)
        self.right_grip_flags = 0x2
        self.right_squeeze = squeeze
        self.right_trigger = trigger
        self.left_trigger = 0.0
        self.left_squeeze = 0.0
        self.button_a = a; self.button_b = b
        self.button_x = False; self.button_y = False
        ang = np.radians(head_yaw)
        self.head = FakePose([0, 0, 0], (0, np.sin(ang / 2), 0, np.cos(ang / 2)))


class FakeServer:
    def __init__(self): self.state = None
    def start(self): pass
    def close(self): pass
    def stats(self): return {}
    def latest(self): return self.state


class FakeArm:
    def __init__(self):
        self.q_ = np.radians([0, 45, -90, 0, 0, 0])
        self.sent = []
        self.pose = np.array([0.30, 0.0, 0.35])
        self.rot = np.array([0.0, 85.0, 0.0])
        outer = self

        class P:
            def GetArmEndPoseMsgs(s):
                class E: pass
                e = E(); e.end_pose = E()
                e.end_pose.X_axis = outer.pose[0] * 1e6
                e.end_pose.Y_axis = outer.pose[1] * 1e6
                e.end_pose.Z_axis = outer.pose[2] * 1e6
                e.end_pose.RX_axis = outer.rot[0] * 1e3
                e.end_pose.RY_axis = outer.rot[1] * 1e3
                e.end_pose.RZ_axis = outer.rot[2] * 1e3
                return e
            def MotionCtrl_2(s, *a): pass
            def EndPoseCtrl(s, *a): outer.sent.append(a)
            def GripperCtrl(s, *a): pass
        self.piper = P()

    def q(self): return self.q_
    def move_j(self, q, speed_pct=0): pass


def run(deltas, squeeze=1.0, cfg=None, ctl_kw=None):
    srv = FakeServer()
    src = QuestCartesianSource(cfg or QuestTeleopConfig(), server=srv)
    arm = FakeArm()
    ctl = CartesianController(arm, dry_run=True, **(ctl_kw or {}))
    ctl.start()
    pos = np.zeros(3); out = []
    for d in deltas:
        pos = pos + np.asarray(d, float)
        srv.state = FakeState(pos, squeeze=squeeze)
        s = src.poll()
        if s.clutch and not s.reset_reference:
            ctl.apply_delta(s.dpos)
        out.append(ctl.target.copy())
    return ctl, out


fails = []

ctl, out = run([[0, 0, -0.02]] * 5, squeeze=0.0)
if not np.allclose(out[-1], ctl.origin):
    fails.append("moved without clutch")

ctl, out = run([[0, 0, 0]] + [[0, 0, -0.002]] * 10)
moved = out[-1] - ctl.origin
if not (moved[0] > 0.015 and abs(moved[1]) < 1e-6 and abs(moved[2]) < 1e-6):
    fails.append("forward mapping wrong: %s" % moved)

ctl, out = run([[0, 0, 0]] + [[0, 0.002, 0]] * 10)
if (out[-1] - ctl.origin)[2] < 0.015:
    fails.append("up mapping wrong: %s" % (out[-1] - ctl.origin))

ctl, out = run([[0, 0, 0], [0, 0, -0.5]], ctl_kw=dict(max_step_m=0.004))
if np.linalg.norm(out[-1] - ctl.origin) > 0.0041:
    fails.append("step clamp failed: %.4f" % np.linalg.norm(out[-1] - ctl.origin))

ctl, out = run([[0, 0, 0]] + [[0, 0, -0.003]] * 400, ctl_kw=dict(max_reach_m=0.10))
if (out[-1] - ctl.origin)[0] > 0.1001:
    fails.append("reach box failed: %.4f" % (out[-1] - ctl.origin)[0])

ctl, out = run([[0, 0, 0]] + [[0, -0.003, 0]] * 400, ctl_kw=dict(min_z_m=0.20))
if out[-1][2] < 0.1999:
    fails.append("z floor failed: %.4f" % out[-1][2])

arm = FakeArm()
c = CartesianController(arm, dry_run=True, max_joint_step_deg=4.0); c.start()
c.check_joints()
arm.q_ = arm.q_ + np.radians([0, 20, 0, 0, 0, 0])
if c.check_joints() or c.aborted is None:
    fails.append("joint watchdog did not fire")

srv = FakeServer(); src = QuestCartesianSource(QuestTeleopConfig(), server=srv)
srv.state = FakeState([0, 0, 0], squeeze=1.0, head_yaw=90.0); src.poll()
v = src._to_robot_base(np.array([-0.01, 0.0, 0.0]))
if not (v[0] > 0.009):
    fails.append("headset-yaw alignment wrong: %s" % v)

srv = FakeServer(); src = QuestCartesianSource(QuestTeleopConfig(), server=srv)
seq = []
for t in [0.0, 1.0, 1.0, 1.0, 0.0, 1.0]:
    srv.state = FakeState([0, 0, 0], squeeze=0.0, trigger=t)
    seq.append(src.poll().gripper_closed)
if seq != [False, True, True, True, True, False]:
    fails.append("gripper toggle wrong: %s" % seq)

# 10. between_clutch: heading re-aims while the clutch is open, freezes while held
srv = FakeServer(); src = QuestCartesianSource(QuestTeleopConfig(), server=srv)
srv.state = FakeState([0, 0, 0], squeeze=0.0, head_yaw=0.0); src.poll()
h0 = src.poll().heading_deg
srv.state = FakeState([0, 0, 0], squeeze=0.0, head_yaw=60.0)
h1 = src.poll().heading_deg                       # unclutched -> should follow
srv.state = FakeState([0, 0, 0], squeeze=1.0, head_yaw=60.0); src.poll()
srv.state = FakeState([0, 0, 0], squeeze=1.0, head_yaw=-40.0)
h2 = src.poll().heading_deg                       # clutched -> should hold
if not (abs(h0) < 1 and abs(h1 - 60) < 1):
    fails.append("heading did not follow the head while unclutched: %.1f -> %.1f" % (h0, h1))
if abs(h2 - 60) > 1:
    fails.append("heading moved while clutched (should freeze): %.1f" % h2)

# 11. continuous mode does follow mid-clutch
cfg2 = QuestTeleopConfig(); cfg2.frame_mode = "continuous"
srv = FakeServer(); src = QuestCartesianSource(cfg2, server=srv)
srv.state = FakeState([0, 0, 0], squeeze=1.0, head_yaw=0.0); src.poll()
srv.state = FakeState([0, 0, 0], squeeze=1.0, head_yaw=45.0)
if abs(src.poll().heading_deg - 45) > 1:
    fails.append("continuous mode did not follow mid-clutch")

# 12. latched mode ignores head motion entirely
cfg3 = QuestTeleopConfig(); cfg3.frame_mode = "latched"
srv = FakeServer(); src = QuestCartesianSource(cfg3, server=srv)
srv.state = FakeState([0, 0, 0], squeeze=0.0, head_yaw=0.0); src.poll()
srv.state = FakeState([0, 0, 0], squeeze=0.0, head_yaw=80.0)
if abs(src.poll().heading_deg) > 1:
    fails.append("latched mode followed the head")

print("checks: %d passed, %d failed" % (12 - len(fails), len(fails)))
for f in fails:
    print("  FAIL:", f)
sys.exit(1 if fails else 0)
