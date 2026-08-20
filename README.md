# RealRobotHT

Bring-up, identification and data collection for a Piper (piperx) arm.

Teleoperation now lives in the **[piperx_teleop](https://github.com/zyc00/PiperxTeleop)**
package (`pip install -e ~/projects/piperx_teleop`); this repo keeps the research
work and the scripts that drive it.

## Teleop and data collection

```bash
python scripts/record_teleop.py --out data/ep01.npz --source quest --unlock-rotation
python scripts/replay_teleop.py data/ep01.npz          # measures replay fidelity
python scripts/record_scripted.py --out data/test.npz  # no human; validates the loop
```

`config/teleop.toml` tunes both. `data/home_pose.npz` is the start pose - keep
**J4 near 0**, or Cartesian range collapses (measured: J4 at 89 deg gave 27 mm of
travel where J4 at 0 gave the full range).

Aliases: `re` (release, arm falls), `reh` (hold), `reo` (open gripper),
`qawake` / `qguard` (headset stay-awake, guardian pause).

## Research (the leader-arm branch)

`piper_ht/` holds the gravity/admittance work, kept because the measurements
were expensive:

- `model.py` URDF gravity model - correct as a model, but the URDF does not
  match this hardware (51 mm mean fit residual)
- `calibration.py` torque-sensing calibration; the SDK under-reports true joint
  torque by 2.4-2.9x and its one coefficient per joint group is wrong
- `gravity_fit.py` gravity model fitted to the real arm, anchored to the URDF
- `admittance.py` admittance control on the position loop
- `next_estimator.py` NEXT-style learned free-space torque model (FACTR 2)

Scripts `01`-`11` and `15`-`17` produced those. The conclusion: **MIT torque
control is inert on firmware S-V1.9-0**, so true gravity compensation is
impossible and a Piper cannot be made into a light leader arm. VR teleop
replaced that approach.

## Diagnostics still useful

- `19_analyze_log.py` - which pipeline stage a bad teleop session died at
- `23_raw_controller.py` - is the headset actually tracking the controller
- `26_tracking_quality.py` - score a headset placement
- `monitor.py`, `park.py`, `release.py`
