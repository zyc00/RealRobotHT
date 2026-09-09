# Leader F/T ablation experiments

## Status and what is actually verified

The scripts implement acquisition, independently confirmed single-trial runs,
static distal-payload calibration, offline analysis, CSV/JSON tables and SVG
condition-by-metric heatmaps/trace plots. Synthetic tests validate the regression,
gravity fit, wrench shift, protocol decoding and condition counts. They are not
robot results. No experiment or sensor-load test has been executed on hardware.

At inspection, all three Ethernet interfaces (`eno1`, `enp6s0f0`, `enp6s0f1`) were
DOWN. Wi-Fi and both CAN interfaces were up. The actual F/T IP, active calibration
counts, mounting transform, distal body and timing are not yet verified. Do not
fill these with guessed defaults to get past the checks.

## Experimental design

These are parameter sweeps/factorial comparisons, not four clean one-factor
ablations. Preserve all other settings, starting pose, grip point, payload,
motion instructions, approximate motion amplitude/speed, and operator between
conditions. Log different participants and blocks separately.

| Study | Factors | Rows |
|---|---|---:|
| Inertia | kappa 0/1/2; friction 0.9, breakaway/damping ON | 3 |
| Friction | friction 0/0.3/0.6/0.9; kappa 2, breakaway/damping nominally ON | 4 |
| Breakaway | beta 0/1; decay 0.05, kappa 2, friction 0.9, damping ON | 2 |
| Damping | translation 0/2 × rotation 0/0.3; kappa 1.5, friction 0.7, breakaway ON | 4 |

The reduced plan's explicit ON defaults are beta **1**, decay **0.05 rad/s**,
damping **2 and 0.3**. These are not falsely presented as parser defaults: the
controller's beta default is **0**, and must be overridden for ON. Change the
plan flags if a different ON setting was intended. “Friction ON” is explicitly
0.9 by default. The manifest is the authoritative condition record. The shorter
decay preserves zero-speed assistance but fades earlier after motion starts;
it is a conservative starting choice, not a experimentally established optimum.
Normal teleoperation launch defaults are not changed by the experiment planner.
New manifests use schema 2 / `reduced_13_rows`; existing manifests are unchanged.

Important interpretation limits:

* Friction scale multiplies breakaway. A zero-friction row has zero effective
  breakaway even when nominal beta=1. This implementation does NOT decouple them.
* There are 13 study rows, but 11 distinct full-controller configurations: the
  all-ON reference appears in the inertia, friction and breakaway studies.
  Damping has its own reference at kappa 1.5 / friction 0.7. `equivalent_key` marks
  this. The runner repeats it within each study rather than silently reusing
  samples. Do not claim those identical reference settings are different effects.
* The damping sweep now includes (0,0), (0,0.3), (2,0), (2,0.3). Zero damping
  even with the reduced kappa 1.5 / friction 0.7 can be unstable; these trials require an additional
  `LOW DAMPING` confirmation. A warning/confirmation does not make them safe.
  Safety-screen with support and an emergency stop before randomized testing;
  retain unsafe/uncompleted conditions as failures rather than bypass safeguards.
* The friction sweeps are not a validation of the friction calibration alone:
  they change residual resistance, breakaway and the shaped residual together.
* Shaping kappa is a controller parameter, not a measured mass ratio. Do not
  label kappa=1 or 2 as an exact apparent inertia without a measured/modelled map.
* Run these validation trials leader-only. The runner never enables a follower.
  A coupled teleoperation/user-study dataset is a separate protocol, with its
  own follower-mode/gains/offset controls.

## Which metrics to use in the paper

| Question | Primary outputs | Important qualification |
|---|---|---|
| Apparent inertia | `apparent_inertia`, fit R², condition number | kg for translation; kg·m² for rotation. Local, axis-specific fit, not a full operational inertia matrix. |
| Friction compensation | axial effort RMS, force RMS, positive translational work, fitted residual Coulomb term | Measured resistance under the complete controller, NOT error of the uncompensated friction model. |
| Breakaway | peak onset effort, effort-to-motion delay, failed-onset count | N or N·m; repeated rest-to-motion trials in both directions, same grip point. |
| Damping | post-release translation/rotation, settling time, interaction work | Requires confirmed release. Reduced motion is evidence of damping behavior, not a passivity proof. |
| Rotation leakage | deg/m, orientation peak error, translation path length | Compute within x/y/z translation trials of every sweep; no separate experiment needed. |
| Validity/safety | cap fraction, guard-active fraction, sensor age/gaps, trips, off-axis motion | Report these; do not silently omit failed/high-assist trials. |

### Corrections to the supplied equations/tables

1. Integrating force avoids direct force-versus-noisy-acceleration regression,
   but does not eliminate friction, damping, gravity error or cross-axis coupling.
   For short, nearly fixed-pose windows use a local empirical model:

   `∫ F_h,i dt = M_app,i Δv_i + B_i Δx_i + C_i ∫ tanh(v_i/v0) dt + b_i Δt`.

   The `d_i Δx_i` term in the screenshot only follows if resistance is linear
   viscous damping `d(v)=d_i v`. It does not represent general Coulomb/Stribeck
   friction. The implementation fits the four coefficients above using
   non-overlapping 0.25 s windows. At least 12 windows, full normalized rank,
   condition number <1000, positive fitted inertia and R²>0.5 are required.
   Include acceleration AND deceleration; constant speed alone cannot identify mass.

2. Correct only the body **distal to the sensing plane**. The existing
   `tool_body.npz` includes the larger joint-6 tool stack and is NOT the sensor's
   distal-payload calibration. The static calibration estimates distal mass,
   COM and six-axis bias with orientation diversity and leave-one-pose-out RMSE.
   It cannot identify rotational inertia: provide a measured/CAD tensor about
   the distal COM in sensor axes for rotational inertia/work estimates.

3. Following the sign convention in the screenshot, analysis uses
   `W_h = sign*W_raw - bias - W_gravity + W_distal_inertia`.
   For force, distal inertia is `m a_COM`, including rotation-induced COM
   acceleration. For torque it includes COM moment, `I α` and `ω × Iω`.
   The scalar `m Δv` translational expression cannot be reused for roll/pitch/yaw.
   Wrench origin shifts include the lever-arm cross product, not rotation alone.

4. All comparisons use axes fixed at the **initial Piper TCP pose**. Translation
   x/y/z use that frame; roll is about tool z (J6), pitch about tool y, yaw about
   tool x. These are axis rotations, not Euler-angle derivatives. Label the axes
   explicitly if the paper uses a different roll convention.

5. `W_FT - W_gravity` does not in general equal `(J^T)^† tau_f` during these
   assisted trials. It contains human input/distal dynamics, while actuator
   compensation and damping change the required input. Therefore Table II's
   proposed MAE/RMSE is NOT emitted as a “friction accuracy” result. Such a table
   needs a separate matched quasi-static protocol and a validated torque balance
   including commanded/actual actuator torque. Compare residual interaction
   effort across the friction sweep instead. Near singularities, a pseudoinverse
   can amplify errors; a small wrench-space MAE is not a universal calibration test.

6. Leakage is `rad2deg(∫ ||ω|| dt) / ∫ |v_i| dt`. The code excludes translation
   paths <=3 cm, uses filtered velocities, and reports orientation peak error.
   Absolute angular speed counts oscillatory rotation even if final orientation
   returns to the start. Report mean x/y/z leakage only after computing each
   direction separately with matched repetitions—not by pooling arbitrary paths.

7. Integral port work is `∫ W_h^T V dt` with common origin/frame and calibrated
   sign. Negative work can be stored-energy return. It is NOT by itself a
   violation or proof of passivity. “Passivity-aware damping” can describe the
   design, but these empirical tests alone cannot certify controller passivity.
   The implemented damping is diagonalized/saturated in joint space, not a
   literal unconstrained Cartesian linear damper.

Inertia estimates are withheld from summary plots when timing is unverified,
off-axis path ratio >0.3, insufficient motion, trips, >50 ms sample gaps, or
guard/core/final-torque caps are active for >5% of a trial. Diagnostic fits remain
in JSON. Missing distal rotational inertia blocks rotational inertia estimates.
Quasi-static torque RMS is explicitly labeled as such, not fully corrected torque.

Processing defaults are a 100 Hz analysis grid, high-rate F/T bin averaging,
and approximately 110 ms cubic Savitzky–Golay position/velocity differentiation.
Report these along with the sensor's actual filter settings; differentiation
and peak metrics remain bandwidth-dependent. Repeat analysis with plausible
filter/delay alternatives before making inertia claims. Onset requires directed
speed >0.005 m/s (or 0.03 rad/s) for 0.15 s; force onset uses 5× baseline noise
or 0.5 N / 0.03 N·m, whichever is larger, sustained for 0.05 s. These thresholds
are protocol choices, not universal physical constants.
`release_confirmed` is a force-consistency heuristic (force <2 N and torque
<0.15 N·m on >80% of samples after 6.5 s), not independent proof of no hand
contact. Settling requires linear speed <0.005 m/s and angular speed <0.03 rad/s
for 0.5 s. Inspect the traces and retain manually annotated release validity.

## Exact workflow

Run commands from `/home/yuchen/projects/RealRobotHT`, using the piperctl Python.
Stop other drag/sensor clients first. Clear the workspace and have the hardware
emergency stop accessible. High friction/shaping/assist combinations previously
showed problematic behavior; a manifest does not establish that they are safe.
Do not continue a condition that pulls unexpectedly or trips repeatedly.

### 1. Connect and inspect the sensor, without robot motion

Power/connect the Net F/T box to the correct Ethernet interface and configure
that interface to the sensor's subnet using the actual network information.
No interface/IP changes have been made automatically. `192.168.1.1` below is a
placeholder from the old B601 reader, NOT a verified address.

```bash
/home/yuchen/miniforge3/envs/piperctl/bin/python examples/ablation_experiment.py sensor-info \
  --ip 192.168.1.1 --out configs/ft_local.json

/home/yuchen/miniforge3/envs/piperctl/bin/python examples/ablation_experiment.py probe \
  --config configs/ft_local.json --out data/ft_probe_01 --seconds 5
```

`sensor-info` reads active units/count scales over HTTP and saves the XML plus a
draft config; only recognized N/N.m units are automatically accepted. It does
not write sensor settings. `probe` checks streaming freshness/status and saves
raw signed counts, sequences and receive timestamps. Stop immediately if loads
are unexpected; do not deliberately overload the sensor to test it.

The wire format and calibration fields follow the manufacturer's
[ATI Net F/T manual](https://www.ati-ia.com/app_content/Documents/9620-05-Net%20FT.pdf).
RDT values need the active counts-per-force/torque scaling; do not assume 1e6.
Use one RDT client at a time. This code never tares the box: a single-pose tare
would hide gravity only at that pose and could invalidate later measurements.

### 2. Verify geometry, signs and timing

Edit `configs/ft_local.json`: fill `T_tcp_sensor` (sensor-to-TCP rigid transform),
confirm `wrench_sign`, and set `geometry_verified=true` only after checking them.
The sensor origin is its defined measurement origin, not necessarily its face.
Account for any transformation already configured in the Net F/T box; do not
apply it twice. A positive calibrated wrench must correspond to positive human
input in the equations above. Check known gentle directional loads and moments.

Record sensor filter/rate and assess sensor/CAN timing/latency independently.
`sensor_delay_s` shifts receive timestamps backwards in analysis. Set
`delay_verified=true` only after a defensible timing check; host receive time is
not a hardware-synchronized acquisition time. Inspect raw gaps and sensitivity
of fitted inertia to plausible delay/filter choices. Without timing verification,
effort/leakage analysis still runs, but apparent inertia is marked unavailable
in aggregate figures. No invented sensor offset or latency is supplied.

### 3. Calibrate the distal body (read-only arm acquisition)

```bash
/home/yuchen/miniforge3/envs/piperctl/bin/python examples/ablation_experiment.py calibrate \
  --config configs/ft_local.json --can can0 --poses 12 --out data/ft_payload_01
```

This command does not enable/disable or reposition the arm. Use the established
safe positioning procedure to choose different orientations. For each sample,
the arm must hold the pose and you must remove hand/contact forces. The script
rejects moving poses and insufficient orientation diversity. Keep the sensor
bias, mounting and distal load unchanged afterward. If the fit fails, the
individual pose files/raw measurements remain available; start a new output
directory for a new attempt. Check fit and held-out force/torque RMSE.

Result: `data/ft_payload_01/payload.json`. For rotational dynamics, add a valid
`inertia_sensor_com_kgm2` 3×3 tensor from CAD/measurement; leave it null otherwise.
Do not substitute inertia of the entire robot/tool stack.

### 4. Generate a manifest and inspect its settings

Pilot, one translation axis, one repeat in each sign:

```bash
/home/yuchen/miniforge3/envs/piperctl/bin/python examples/ablation_experiment.py plan \
  --out data/ablation_reduced_pilot.json --axes x --repeats 1 \
  --on-breakaway 1 --on-decay 0.05 --on-damping 2 0.3 --friction-on 0.9
```

This contains 26 short trials (13 rows × two signs), 5.2 minutes recording. It is a
randomized manifest, not an automatic batch of robot motions. Inspect it and
start with lower-assist conditions; randomization belongs AFTER a safety pilot.
For repeated translation data use `--axes x y z --repeats 5`: 390 trials,
78 minutes recording alone. With an assumed 20–40 seconds for restart,
repositioning and confirmation per trial, allow roughly 3.5–5.6 hours, plus
sensor calibration/rest. Three repeats gives 234 trials / 46.8 minutes recording
(roughly 2.1–3.4 hours at that same overhead). These are planning estimates,
not measured timings. Five repeats on all six axes gives 780 trials / 156 minutes
recording. Split into balanced blocks; repeats are not a statistical power analysis.
Use a distinct `--participant` and seed/manifest per participant.

Timing comparison on the SAME basis (x/y/z, two signs, five repeats, 12 s/trial):

| Design | Trials | Recording only | With 20–40 s overhead per trial |
|---|---:|---:|---:|
| Original 44 rows | 1320 | 264 min / 4.4 h | 11.7–19.1 h |
| Reduced 13 rows | 390 | 78 min / 1.3 h | 3.5–5.6 h |

The previous 4.4-hour figure excluded setup; comparing it with the reduced
setup-inclusive estimate was misleading. The reduced grid removes about 70.5%
of trials. Damping gain magnitudes do not change the number/duration of trials.
Neither overhead estimate is measured hardware timing. Existing manifests
retain their original settings; generate a new file to use kappa 1.5 / friction
0.7 in the damping rows.

```bash
/home/yuchen/miniforge3/envs/piperctl/bin/python examples/ablation_experiment.py plan \
  --out data/ablation_reduced_full.json --axes x y z --repeats 5 \
  --on-breakaway 1 --on-decay 0.05 --on-damping 2 0.3 --friction-on 0.9

/home/yuchen/miniforge3/envs/piperctl/bin/python examples/ablation_experiment.py list \
  --plan data/ablation_reduced_pilot.json --study inertia --limit 10
```

`list` displays actual randomized trial IDs/settings; select a screened condition
from that output. Do not assume `trial_0000` is an appropriate first physical test.

### 5. Run one named trial

```bash
/home/yuchen/miniforge3/envs/piperctl/bin/python examples/ablation_experiment.py run \
  --plan data/ablation_reduced_pilot.json --trial YOUR_SELECTED_TRIAL_ID \
  --config configs/ft_local.json --root data/ablation_reduced_pilot_runs
```

The script displays that trial's settings/axis/direction and requires `RUN`.
The normal guarded startup may require `MOVE`, then Enter to start drag. Both
F/T and robot data are recorded; a sensor fault/stale stream aborts torque drag.
The UI displays the run but parameter changes are locked. The controller emits
phase cues: rest 0–2 s; directed movement 2–6 s; release 6–12 s. Release only when
the arm can safely remain clear of obstacles; emergency-stop unsafe behavior.
At release avoid further hand contact, or release metrics will be invalid.

Use approximately 3–10 cm translation strokes (adequate for the metric but
within a clear local workspace), or modest isolated axis rotations. Keep grip
point and trajectory profiles comparable. Include speed rise/fall; pushing at
constant speed throughout makes inertia unidentifiable. Both positive/negative
trials matter for directional friction. For an actual controlled force profile,
use a characterized external fixture, not a hand push labeled as a known force.

Each trial is immutable; existing output paths are refused. After a failed
trial, retain it as a failure and create a new run-root/repeat, not an overwrite.
Trial IDs are explicitly selected from the manifest; do not assume 0000 is the
least aggressive condition. No follower is commanded by the experiment runner.

### 6. Analyze and view figures

```bash
/home/yuchen/miniforge3/envs/piperctl/bin/python examples/analyze_ablations.py analyze \
  --root data/ablation_reduced_pilot_runs --payload data/ft_payload_01/payload.json \
  --out data/ablation_reduced_pilot_figures
```

Outputs:

* `metrics.csv/json`: each trial, validity/failure reason, fit diagnostics and metrics.
* `summary.csv/json`: condition/axis means, valid n and trial-bootstrap 95% intervals
  when at least three trials contribute. These are NOT participant-population CIs.
* `inertia_x_heatmap.svg`, `friction_x_heatmap.svg`, etc.: each condition × metric,
  values/valid n in cells, gray for unavailable. Color is normalized per metric
  column; it is not a confusion matrix and cannot be compared across columns.
* `trial_XXXX_trace.svg`: force and velocity traces with separate unit axes.
* `trial_XXXX_processed.npz`: corrected wrenches/velocities/poses for custom figures.

Open SVG files in a browser or editor. No matplotlib/pandas installation is
required; the existing environment has NumPy/SciPy. Controller logs and CSV
streams remain independent of these derived outputs. Per-trial raw data include
sensor counts/status/sequences/gaps, host wall+monotonic time, robot feedback
time, q, filtered velocity, gravity/torque commands, observer residual,
breakaway torque, safety torque, shaping alpha, trips and cap flags. Manifests
include source/calibration hashes. Asynchronous CSV writes flush every 100 rows;
normal exits preserve complete files, but hard power loss may lose the tail.

For a no-hardware analysis/figure smoke test:

```bash
/home/yuchen/miniforge3/envs/piperctl/bin/python examples/analyze_ablations.py demo \
  --out data/SYNTHETIC_ablation_demo
```

Every result from this command is synthetic. Do not put it in the paper as a
robot experiment. The synthetic repeated trials are identical by design and
their intervals do not represent experimental uncertainty.

## Suggested paper figures and statistics

Use one heatmap per sweep/axis for exploration. For the paper, choose compact
panels: inertia versus kappa at friction 0.9; effort versus friction at kappa 2;
breakaway peak versus beta at fixed decay 0.05; and a 2×2 translational/rotational
damping grid for settling time and post-release motion. Show points
for every trial and uncertainty over independent trials/participants. Put
leakage beside effort to expose the effort-versus-orientation tradeoff. Report
failed trials and saturation/guard fractions rather than hiding them.

The supplied SVG heatmaps/traces and summary tables are implemented. Factorial
publication styling and mixed-effects participant statistics remain subsequent
analysis choices. The summary currently pools direction within each condition/
axis; inspect direction-specific trial CSVs before pooling asymmetric behavior.
Use participant-level aggregation or mixed-effects models for a user study,
counterbalance order, account for repeated measures and multiple comparisons,
and never count 100 Hz samples as independent repetitions. Add a dedicated
between-configuration user-study design if making human performance claims.

## UMI comparison

[UMI](https://umi-gripper.github.io/umi.pdf) is a handheld demonstration interface,
not an actuated robot arm with the same dynamics. Its original results do not
supply your apparatus's force–acceleration benchmark. Do not use its task speed
as an apparent-inertia measurement.

A defensible additional comparison would instrument the actual handheld UMI at
the same hand interaction point, with matched sensor/fixture/load, task axes,
motion profile and processing. Report mass/COM/inertia of the entire handheld
device and the added measurement fixture separately, plus measured effort,
apparent inertia (where identifiable), and leakage. Gravity support, friction,
workspace constraints, camera/handle ergonomics and human strategies differ.
Call this a matched-task effort/apparent-inertia benchmark, not proof the robot
is universally “as light as UMI.” The current scripts collect Piper data only;
no UMI pose acquisition adapter or UMI physical trial has been implemented.
