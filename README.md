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
