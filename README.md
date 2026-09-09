# RealRobotHT

Teleoperated data collection for a Piper (piperx) arm.

Teleoperation itself lives in **[piperx_teleop](https://github.com/zyc00/PiperxTeleop)**:

```bash
pip install -e ~/projects/piperx_teleop           # keyboard
pip install -e "~/projects/piperx_teleop[quest]"  # + Quest 3
```

## Record

```bash
python examples/record_teleop.py --out data/ep01.npz --source quest --calibrate
python examples/record_teleop.py --out data/ep01.npz --source keyboard
python examples/record_teleop.py --out data/ep01.npz --source scripted   # no human
```

One row per control tick: `action` (commanded pose, replayable), `intent` (what
the operator asked for, before limiting), `clutch` (demonstration vs
repositioning), proprioception, and four timestamps.

## Replay

```bash
python examples/replay_teleop.py data/ep01.npz --dry-run
python examples/replay_teleop.py data/ep01.npz
```

Homes to the recorded starting joints - the firmware's IK resolves from the
current configuration, so starting elsewhere can put the same end-effector path
on a different joint solution - then replays and reports the error. Measured
0.5 mm mean over 1590 ticks.

## Config

`config/teleop.toml` is loaded by the record script (rotation is enabled there by default) and passed into the teleop
objects; the package never reads a config file itself.

The two keys you will actually tune are `workspace.max_reach` and
`workspace.min_z`. **`max_reach` is measured from where the session starts**, so
to work near a table, start the session low rather than opening the box wide.
Set `min_z` just above the surface as a collision guard.

## Start pose

`data/home_pose.npz` holds the joint pose each session begins from. Keep **J4
near 0**: measured, J4 at 89 deg gave 27 mm of vertical travel before the arm
stalled, while J4 near 0 gave full range in every direction.

## Helpers

Installed with the package, so they work from any directory:

```bash
re      # piperx-release        motors + gripper off - THE ARM WILL FALL
reh     # piperx-release --hold freeze in place
reo     # piperx-release --open open the gripper only
qawake  # keep the headset awake off-head
qguard  # pause the headset boundary system
```

## Drag mode with the F/T tool (torque mode, gravity + friction)

Since 2026-09-08 an ATI Nano25 F/T sensor and two adaptors sit past the joint-6
flange. Gravity comp models everything past joint 6 as ONE rigid body and reads it
from a tool file; friction comp reads a calibrated Stribeck model. Both `piperctl`:

```bash
python examples/tool_id.py prior                    # CAD + datasheet -> data/tool_prior.npz (holds)
python examples/tool_id.py torque && python examples/tool_id.py fit   # refine mass in t_ff units
python examples/friction_cal.py run                 # 6 poses, static + multi-speed kinetic sweeps
python examples/friction_cal.py fit                 # -> data/friction_model.npz
python examples/drag_mode.py --tool data/tool_body.npz --gravity data/gravity_cal.npz --friction data/friction_model.npz
```

Two facts that cost a day: the **reported joint effort cannot weigh the tool**
(each joint's current coefficient is unknown - slopes 0.28/0.29/0.82/1.16 on
J2-J5 against the model - so mass and coefficient trade off; `tool_id.py check`
only confirms the model's shape), and the **firmware MIT kp/kd are not in
N.m/rad**, so the kinetic friction sweep runs a host-side PD in `t_ff`. The
static sweep brackets breakaway in both directions: friction is the
half-difference, the gravity residual is the half-sum (free model check).
`--fric-scale` (default 0.8) must stay below 1 or joints creep.

## Per-joint gravity bias (`gravity_cal.py`)

Both breakaway brackets (friction_cal static rows, tool_id torque) record `(u+ + u-)/2`, the
torque the model is missing at that pose in t_ff units. After the tool fit, what is left on
J1 and J3 is a CONSTANT per joint (J3 -0.37 N.m, J1 +0.19, no correlation with load or
pose) - which for a Coulomb model is the same thing as direction-asymmetric friction, and
is compensated identically: `tau_ff = scale * g(q) + bias` (b601's g_scale/g_bias).

```bash
python examples/gravity_cal.py        # reads friction_cal.csv + tool_torque.npz -> data/gravity_cal.npz
python examples/drag_mode.py --tool data/tool_body.npz --gravity data/gravity_cal.npz --friction data/friction_model.npz
```

Because the bias absorbs the asymmetry, friction_cal's static levels are symmetric. Re-run
gravity_cal after any new friction_cal or tool_id data.

### Separate static and moving residuals

The legacy gravity calibration above fits breakaway midpoints, which mix gravity
error with directional static friction. That midpoint need not remain valid in
motion. Compare the two regimes for all six joints without overwriting calibration:

```bash
python examples/separate_bias.py --measured-with data/tool_prior.npz --out data/bias_separation_new.npz
```

Specify the tool model actually used during acquisition; old CSVs do not record it.
The script pairs opposite kinetic sweeps, rejects missing or mismatched motion,
fits static and moving scale/bias independently with equal weight per pose, and
reports their difference at matching poses. It saves diagnostic arrays and a
Markdown report, including sample counts and scatter. Existing outputs are not
overwritten. These files are not `--gravity` inputs: moving residuals still mix
gravity, friction asymmetry and dynamic errors. Legacy sweep-centre positions
make tool-model conversion approximate. Current results are in
`data/bias_separation.md`; static and moving biases differ notably on J1–J3.

## B601-aligned drag on Piper

`examples/drag_mode.py` now runs only the B601-aligned implementation, using an
unchanged snapshot of `b601_teleop/b601/balance.py`. Start with assistance off
and adjust inertia shaping and friction compensation from the panel:

```bash
/home/yuchen/miniforge3/envs/piperctl/bin/python examples/drag_mode.py \
  --tool data/tool_body.npz --gravity data/gravity_cal.npz \
  --friction data/friction_model.npz --balance 0 --fric-scale 0 \
  --serve --log data/b601_reference_01.npz
```

Add `--dry-run` to validate the models without connecting to CAN. The reference
panel at http://127.0.0.1:8731 controls shaping, friction, Cartesian damping and
B601 breakaway. `--help` lists this mode's options. It rejects
Piper-only options such as `--balance-fr`, `--balance-coupling`, `--breakaway`
and `--balance-relief`, rather than silently changing reference behavior.

Preserved from B601: momentum observer excluding friction from beta, friction
relief added to r_net, full eigen-clipped K (no row normalization), rest-bias
learning, 25–40 condition-number fade, torque clipping, runaway detector,
30 ms position-difference velocity filter, and previous-command PD reconstruction
using midpoint velocity. Default reference CLI policies are mass 1.8 kg,
pitch/yaw inertia 0.06, roll inertia 0.0005, observer 3 Hz, Cartesian damping
2 / 0.3 with knee 0.08, velocity taper 0.03, soft intent floor 0.5, detent and
breakaway off. The loop target is 100 Hz, like the B601 config. Actual timing
still depends on Piper's runtime and feedback transport.

Hardware adaptations are explicit: Piper URDF/tool/calibrated friction, a proper
rotation of tool-frame Jacobian rows (Piper roll-z to B601 roll-x), Piper's CAN
runtime/watchdog/position-hold exit, and Piper-specific torque limits. The dynamics
use URDF-only inertia (`rotor=0`) as B601 does. Combined assistance caps are
[0.6, 1.5, 1.0, 0.4, 0.3, 0.2] N·m; total feed-forward caps are
[8, 10, 8, 3, 3, 3] N·m. Limits are recorded before observer command accounting.
They can restrict the reference output differently from B601's actuator limits.

**`--kd` means raw firmware MIT damping gains**, not host-side N·m·s/rad. Defaults are all zero, with this port limiting inputs to
0..0.5. B601's firmware command policy is preserved, but Piper gain units have
not been calibrated; nonzero kd therefore makes physical damping and its observer
reconstruction uncertain. Do not copy Piper mode's `--kd ... 4 ...` command here.
The Cartesian damper remains active with zero firmware kd.

Gravity always uses the file's constant legacy `scale`/`bias` keys; scheduled
static/moving corrections and encoder-directed breakaway are bypassed. Optional
`--balance-breakaway` is B601's single scalar observer-directed term on all joints,
rather than the removed encoder-directed proximal-only vector. No additional Piper friction feed-forward
or host damping is added. Launch defaults for kappa/friction are zero for initial
validation, rather than B601 CLI's assisted defaults.

The superseded controller, scheduled-bias adapter, encoder-directed breakaway,
and old panel were removed from the active source tree. Their exact source backup
is `data/legacy_drag_sources_before_b601.tar.gz`. Calibration files, historical
logs and unrelated examples remain intact. `--controller b601` is accepted for
old launch commands; there is no alternate controller. Shared `piperx_teleop`
package code remains available to other projects; this script uses only its
hardware runtime and model support.

Logs have an explicit `controller=b601-reference` schema, per-tick live settings,
residual, combined assistance and commanded firmware gains. `log_analyze.py`
recognizes it. Snapshot SHA256:
`888329aa0fee08c9b062dc5f895763397337e86d001f779065f6ef4dbbeac471`.
Tests replay rest, movement, reversals, saturation and gain toggles against the
original B601 class. Numerical parity does not establish hardware stability.

## Leader/follower joint mirroring

For leader-only F/T validation, randomized ablation manifests, acquisition,
distal-payload calibration and plots, see [the experiment run guide](docs/EXPERIMENTS.md).
Start with its sensor/network checks; experimental condition lists do not certify
high-assist settings as safe, and synthetic demo figures are not robot results.

The leader panel has one **Breakaway fraction (all leader joints)** slider,
0–1, initialized by `--balance-breakaway`. It sets the same fraction on all six
joints; 1 means each joint's full calibrated breakaway term, still multiplied
by friction scale and gates and subject to existing torque caps. It is not a
shared budget divided among six joints. Per-joint UI/API controls were removed;
per-joint torque and saturation readouts remain available.

`joint_breakaway.py` extends the archived B601 recurrence with the fraction
vector and telemetry; the original snapshot remains unmodified. Equal fractions
are tested for numerical equivalence. The panel reports `breakaway_torque`
(requested direct term before shaping/total caps), `core_saturated` (total core
extra torque over its cap), and `feedforward_saturated` (gravity plus guarded
extra over the final cap). These flags are software saturation, not measured
motor saturation. Logs add a `breakaway` array with per-joint fractions, requested
torques, pre-cap core outputs and saturation flags. Inertia shaping still couples
joints through its matrix.

Follower J6 now defaults to **leader J6 -45°** in both MOVE_J and MIT modes.
Override with `--follower-j6-offset-deg 0` to restore direct mirroring, or `45`
to reverse the correction. This is a joint-angle offset, not a gripper-opening
command. Targets still slew; there is no immediate 45° target step. The leader
guard uses the follower's model/firmware envelope shifted by the negative offset.
For the current limits this reduces leader J6 to approximately [-74°, +119°],
while follower J6 remains [-119°, +119°]. Startup uses each arm's own guard;
measured follower angles are not offset again. Logs retain raw leader angles
and record `joint_offset_rad`; telemetry includes the mapped target. No hardware
motion was performed to verify the physical correction direction.

### Optional experimental MIT follower

Add `--follower-mode mit --follower-kp 10 --follower-kd 0.8` to select MIT PD
position tracking instead of the default MOVE_J follower. Both arms' confirmed
startup positioning still uses MOVE_J. A separate `MIT` confirmation is required
before starting the run, even with `--yes`. The follower seeds its own measured
pose, verifies fresh MIT-mode feedback, then follows at a nominal 100 Hz.

The same `--serve` panel exposes **Follower MIT Kp** (1–30) and **Follower MIT Kd**
(0.1–2), each shared across all six joints. These are raw firmware gain fields,
not calibrated physical units or certified-safe bounds. Live changes slew at
5 kp units/s and 0.5 kd units/s. The target-speed slider remains active; firmware
speed percentage is disabled because it does not govern MIT PD tracking.
Applied kp/kd are saved in the follower log's `applied_gains` array.

MIT tracking commands zero desired velocity and zero feed-forward torque; it
does **not** reuse the leader's tool/gravity calibration. Without follower-specific
gravity compensation, static position error supplies load-support torque and
sag is possible. Support the follower load during initial testing, use a low
target speed, and keep the hardware emergency stop accessible. This mode has
offline tests only, not hardware validation. Do not assume increasing kp resolves
the prior MOVE_J non-response.

The MIT follower trips immediately for >5° command tracking error, actual pose
outside the shared guard, mode loss, firmware faults, disabled motors, stale
feedback, or a >100 ms control-loop gap. Stop requests MOVE_J position hold;
delivery and physical stopping cannot be guaranteed if CAN/control fails. These
software checks cannot limit firmware PD torque directly, stop a dead host, or
detect collisions. Keep the operating envelope clear.

### Default MOVE_J follower

The `--serve` panel includes live **Follower target ramp (rad/s)** and
**Follower firmware speed (%)** sliders, initialized from `--follower-speed`
and `--follower-speed-pct`. They are disabled without a follower. Changes are
applied by the follower control loop, not the web-server thread; firmware speed
updates preserve the mounting configuration. These controls affect mirroring,
not the slower confirmed startup moves. Applied speeds are recorded per sample
in the follower log's `speed_settings` array. Increase speeds cautiously; the
joint-limit and tracking-error protections remain unchanged.

Before paired drag starts, both arms are offered concurrent, independent
position-only startup moves if either is near/outside the guard. Each arm's
target clips only its own boundary-near joints
to 2 degrees inside the hard guard (normally J2 +3°, J3 -3°).
This minimizes the startup move; the leader can start within the soft-resistance
zone, so inward resistance may be felt after entering drag mode. Operating
limits and tracking watchdog thresholds are not widened.
The preview shows both targets and requires typing `MOVE`, even with `--yes`.
Both arms move during this step, but mirroring remains off. The normal Enter
prompt follows only after both reach their own targets. Their poses need not
be identical: after Enter, the follower approaches the leader as before.
Startup uses MOVE_J at 10% firmware speed and a 3°/s target ramp, pauses the
ramp when tracking error reaches 1°, and aborts on stale feedback, disabled
motors, mode/fault errors, excessive tracking error, or timeout. A parked pose
up to 5° outside the nominal envelope first targets the nearest valid boundary;
this initial correction is not covered by the 3°/s target-ramp bound. Larger
excursions are refused. Cleanup requests a best-effort position hold. Clear the
entire path of both arms before confirming; joint limits do not check collisions.

Leader/follower runs now query all six joint limits from **both** arms before
starting, and intersect them with the conservative model limits. Missing or
invalid replies prevent startup. A software guard stops the run 1 degree inside
that common envelope. Over the preceding 8 degrees, an external safety layer
tapers outward non-gravity assistance and adds bounded inward spring/damping
torque. The B601 reference core is unchanged; its observer is told the actual
post-guard, post-cap command. The startup output prints the guarded ranges.
No additional CLI flags are required. Reposition inside the printed range before
restarting after a limit trip; do not bypass the checks or widen limits to clear it.

The panel state exposes `limit_proximity` (0: clear, approaching 1: boundary)
and `safety_delta`; logs save these separately in `safety`. The leader also
checks follower feedback freshness during active torque control. Existing
position-mode, tracking-error and feedback watchdogs remain enabled.

This is **not a physical constraint or certified safety system**: resistance is
torque-limited and a human can overpower it. A trip requests position hold, not
motor disable, and a failed CAN bus cannot guarantee that hold is received.
Joint limits do not detect self-collisions, table/obstacle collisions, tool
collisions, or guarantee stopping distance. Clear both workspaces, start slowly,
and keep the hardware emergency stop accessible. Hardware motion testing is
still required before relying on the guard.

The discovered adapters are `can0` (leader, USB `1-11.1:1.0`) and `can1`
(new follower adapter, USB `1-3:1.0`). Interface names should be rechecked after
replugging/rebooting. Both need 1 Mbps. The new adapter was initially down:

```bash
sudo ip link set can1 up type can bitrate 1000000
/home/yuchen/miniforge3/envs/piperctl/bin/python examples/drag_mode.py \
  --can can0 --follower-can can1 \
  --tool data/tool_body.npz --gravity data/gravity_cal.npz \
  --friction data/friction_model.npz --balance 0 --fric-scale 0 \
  --serve --log data/leader_follower_01.npz
```

Before pressing Enter, clear both arms' paths: the follower will approach the
leader's current pose. The leader uses B601-aligned torque drag; the follower uses
normal MOVE_J position control and absolute J1–J6 targets (no relative offset or
joint sign changes). The firmware follows streamed joint targets at 50 Hz with
a default 0.5 rad/s per-joint command slew limit and 20% position speed. These
limits deliberately cause lag during fast motion; matching is not instantaneous.
Use `--follower-speed` and `--follower-speed-pct` to configure them. The follower
needs no F/T sensor or tool/gravity calibration. Different tools mean equal joint
angles do not imply identical tool-tip positions. Grippers are not mirrored.

Feedback older than 150 ms on either arm, disabled follower motors, unreachable
leader targets, persistent follower tracking error, or loss of position mode
stops mirroring and requests position hold on both arms. Ctrl-C also holds both;
it does not disable motors. If follower feedback is stale, its hold request uses
the last commanded target. A failed CAN link cannot guarantee delivery of hold
commands; retain access to the physical stop. Startup itself is speed-limited.

The panel includes follower target/actual angles and tracking error. `--log`
also saves `<stem>.follower.npz` with leader, follower and commanded joint angles.
`--dry-run` validates software/models without CAN access, so it does not verify
that a live follower is responding. At startup a silent follower is restored to
normal motion-output communication with default CAN IDs (`MasterSlaveConfig`
0xFC), then given up to two seconds to produce fresh joint feedback. This step
does not enable motors or send position targets. Startup still fails if feedback
does not recover. On this follower, the restore recovered 200 Hz joint feedback;
position mirroring has been verified offline but not yet through a hardware run.
Follower position mode preserves its configured mount (`installation_pos=0`)
and is selected once at startup, then only joint targets are streamed. Reasserting
the mount on this firmware paused joint feedback for over 200 ms in a position-hold
test; leaving it unchanged kept feedback age below 5 ms. The runtime stale-feedback
limit remains 150 ms. No follower load or mounting parameters are rewritten.
Leader MIT-mode entry likewise preserves its mount setting. Before starting the
torque loop it requires a post-switch MIT status and fresh joint feedback stable
for 50 ms, with a 2.5 s startup timeout. Cached pre-switch feedback cannot release
the follower into tracking. Startup readiness is separate from the unchanged
150 ms runtime stale-feedback watchdog.

## Admittance demo (not a working teleop mode)


```bash
python examples/admittance_demo.py --light
```

Push the arm and it follows your hand. Kept as a demonstration only. Force
sensing works well (SNR 47-135 on J2/J3), but it never became usable:

- breakaway ~4.2 N against ~1.3 N of real joint friction - admittance must
  INFER a push from torque and cannot tell your hand from the joint sitting
  somewhere in its friction band
- the deadband is floored by gravity-model error, and the vendor URDF does not
  match this arm (51 mm rigid-fit residual)
- it drifts; the zeroing, hysteresis and drift guards reduce but do not remove it

Unfixable in software: **MIT torque control is inert on firmware S-V1.9-0**, so
real gravity compensation is impossible and this was the workaround. VR teleop
replaced the need for it.

Runs with no setup, using the analytic URDF gravity model and wider deadbands.
For the better fitted model, the identification sweep is in the history at
`380ce14` (`scripts/06_identify.py`, `07_fit_gravity.py`).

## NEXT: learned free-space torque model

A NEXT-style estimator (FACTR 2, [arXiv 2606.12406](https://arxiv.org/abs/2606.12406))
predicts the torque a **contact-free** arm should be drawing, so the residual
against measured torque is the external torque. It replaces the analytic gravity
model, whose error was 6-47x the arm's own repeatability - which is what forces
the heavy admittance deadbands.

```bash
python examples/next_collect.py                  # ~10 min of arm motion, DO NOT TOUCH IT
python examples/next_train.py --no-cmd           # ~20 s on the GPU
python examples/next_fit_deadband.py data/next_model.pt data/next_deadband.npz
python examples/admittance_demo.py --light       # picks the model up automatically
```

Measured on this arm: the learned model beat the analytic one on every joint
(J2 0.37 -> 0.30, J4 0.28 -> 0.14 reported N.m). But most of the practical gain
came from a diagnostic rather than the network - the error scales with **overall
arm speed**, not each joint's own, so a speed-dependent deadband cut J4's from
1.63 to 0.17 and J2's from 0.88 to 0.41.

Two things worth knowing before trusting it:

- **`--no-cmd` matters.** The paper feeds `q_cmd - q` as an input, which is right
  when `q_cmd` comes from a human teleoperator. Fed to an admittance loop it
  closes a feedback path, since `q_cmd` is then that loop's own output, and the
  arm drifts. Accuracy without it was unchanged.
- **58% of poses in our collection had near-duplicates**, because the logger
  repeats its trajectory programme, so validation numbers are optimistic.

## Notes

- Hand-dragging the arm puts it in teach mode, after which every command is
  silently discarded. Press the teach button on the arm; it cannot be cleared
  over CAN.
- Quest 3 controllers are tracked optically. Out of the headset's view the IMU
  dead-reckons a valid-looking pose that does not follow your hand; the session
  warns when it detects this.
- `can0` down: `sudo ip link set can0 up type can bitrate 1000000`

The gravity-compensation and admittance research that preceded this (and the
finding that MIT torque control is inert on firmware S-V1.9-0) is in the git
history, at commit `380ce14`.
