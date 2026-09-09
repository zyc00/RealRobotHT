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
