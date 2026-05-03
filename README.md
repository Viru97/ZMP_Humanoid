# ZMP HUMANOID ROBOT CONTROL

A whole-body control pipeline for the Unitree G1 humanoid robot using MuJoCo simulation.

## Project Status

| Phase | Description | Status | Notes |
|-------|-------------|--------|-------|
| **Phase 1** | Settling (default pose hold) | ✓ Working | Natural standing under gravity |
| **Phase 2** | Balance Acquisition | ✓ Working | CoM shift toward support foot center |
| **Phase 3** | Stability Hold | ✓ Working | Verify balance before motion |
| **Phase 4** | ZMP Preview Control | ✓ Working | Lateral sway via safe_sway fallback (preview-IK disabled) |
| **Phase 5** | Single-Step Walking | ⚠ In Progress | Swing motion implemented; gait unstable |

## Architecture

### Control Flow
```
main.py
  ├── VisualizedPhaseManager (orchestration)
  │   ├── G1RobotModel (kinematics)
  │   ├── WholeBodyIK (inverse kinematics)
  │   ├── JointController (actuator dispatch)
  │   ├── StatusMonitor (diagnostics)
  │   └── ZMPPreviewController (Phase 4, not used)
  └── MuJoCo Simulation
      ├── Model: assets/g1.xml
      ├── Physics: 0.001s timestep, 1000 Hz
      └── Control: 0.01s loop, 100 Hz
```

### Key Files

| File | Purpose |
|------|---------|
| `scripts/main.py` | Entry point; phase invocation and visualization |
| `scripts/phase_manager.py` | Phase logic, control loop, diagnostics |
| `scripts/whole_body_ik.py` | Iterative Gauss-Newton IK solver |
| `scripts/g1_robot_model.py` | Robot structure, foot IDs, kinematics |
| `scripts/joint_controller.py` | Actuator command dispatch (position servo) |
| `scripts/config.py` | Centralized tuning parameters |
| `assets/g1.xml` | MuJoCo model definition |

## Running the Pipeline

```bash
python scripts/main.py
```

The robot will:
1. Settle into standing pose (2 seconds)
2. Shift CoM to center of support polygon (2 seconds)
3. Hold stable posture (2 seconds)
4. Perform lateral sway (5 seconds)
5. Attempt a forward step (1.5 seconds)

A MuJoCo viewer window opens showing real-time visualization. Close the window to exit.

## Diagnostics Output

### CoM and Balance Tracking
```
[PHASE  ] tick=  100 | h=0.6951 | CoM=[0.032,0.000,0.695] | FC=[-0.000,0.000] | xy_err=0.0320
```
- `h`: CoM height (meters)
- `CoM`: Center of mass position [x, y, z] in world frame
- `FC`: Foot center position (midpoint of both feet)
- `xy_err`: Horizontal distance from CoM to foot center (stability metric)

### Phase 5 Joint Diagnostics
```
[STEP JOINTS] name                 q_cur     q_tgt     q_err     qd       cmd      tau_act
[STEP JOINTS] right_hip_roll_joint +0.1086  +0.1327  +0.0241  +0.4215  +0.1327  -4.38
```
- `name`: Joint name
- `q_cur`: Current joint angle (radians)
- `q_tgt`: IK target angle (radians)
- `q_err`: Tracking error (q_tgt - q_cur)
- `qd`: Joint velocity (rad/s)
- `cmd`: Actual control command sent (position servo mode)
- `tau_act`: Estimated actuator torque (Nm)

**Key observation**: When `tau_act` becomes very large (> 20 Nm), the joint is hitting a limit or the commanded motion is infeasible.

### Phase 5 Foot Diagnostics
```
[STEP FEET] lf_act=[+0.026,+0.119,+0.018] lf_tgt=[+0.026,+0.119,+0.018]
[STEP FEET] rf_act=[+0.026,-0.119,+0.018] rf_tgt=[+0.026,-0.119,+0.018]
```
- `lf_act`, `rf_act`: Actual left/right foot positions [x, y, z] (meters, world frame)
- `lf_tgt`, `rf_tgt`: IK target foot positions

**Interpretation**: If actual and target diverge significantly, the swing foot is not tracking the commanded trajectory.

### Support Margin
```
[STEP] support_margin=0.0370m
[STEP] support_margin=-0.0228m (WARNING: support margin became too negative!)
```
- Approximate distance from CoM projection to support polygon edge (meters)
- Positive: inside support region (safe)
- Negative: outside support region (unstable)
- **Note**: This is a conservative heuristic; does not directly cause step failure

## Phase 5: Single-Step Design

### Kinematics
- **Weight Shift (0–35% of duration)**:
  - CoM interpolates from center of foot rectangle toward right foot
  - Smoothstep interpolation for smooth velocity profile
  - Both feet remain planted

- **Swing (35–75% of duration)**:
  - Left foot follows sinusoidal arc: `z = swing_height * sin(π * progress)`
  - Horizontal: linear interpolation from current to landing position
  - CoM stays over right foot (support foot)
  - Right foot planted and constrained

- **Landing Transition (75–100% of duration)**:
  - CoM begins moving back toward center of new foot rectangle
  - Left foot lands (z returns to ground level)
  - Smooth transition to double-support phase

### Control Strategy
- **IK Targets**:
  - `com_target`: Desired CoM position [x, y, z]
  - `lf_pos_target`: Left foot desired position (updated dynamically during swing)
  - `rf_pos_target`: Right foot position (constant, planted)

- **IK Solving**:
  - Gauss-Newton iteration, 12 iterations per control step
  - Foot position weighted 200x higher than CoM position
  - Foot orientation weighted 100x
  - CoM weighted 60x
  - Integration step: 0.001s per iteration (fixed, conservative)

- **Position Servo**:
  - All joints are position-controlled via MuJoCo built-in PD
  - kp=500 Nm/rad, kd=43 Nm*s/rad (set in model XML)
  - Control command is desired joint angle; servo generates torque internally

### Parameters (from `config.py`)
```python
STEP_LENGTH = 0.03      # [m] Target step length (forward)
SWING_HEIGHT = 0.02     # [m] Maximum foot lift
FOOT_LENGTH = 0.16      # [m] Foot size (support rectangle)
FOOT_WIDTH = 0.08       # [m] Foot size (support rectangle)
```

## Known Issues and Failure Modes

### Phase 5 Instability
**Symptom**: Robot collapses during swing phase (tick ~100 of 150)

**Observed Behavior**:
- Right hip roll joint torque spikes to -21 Nm (> mechanical limit)
- CoM height drops from 0.695 m to 0.673 m
- Base orientation tilts excessively (tilt > 0.60 rad)
- Left foot swing target diverges from actual position
- CoM drifts laterally (Y coordinate becomes -0.254 vs target ~0.0)

**Root Causes (Hypotheses)**:
1. **Hip roll saturation**: Joint torque limit (150 Nm nominal, ~50 Nm safe) insufficient for lateral stabilization
2. **Lateral CoM drift**: Weight shift or IK CoM target biasing not strong enough; CoM overshoots support foot
3. **Swing foot tracking lag**: Position servo cannot keep up with aggressive swing trajectory under load
4. **IK infeasibility**: Desired CoM + foot positions may be geometrically infeasible given joint limits

### Partial Solutions Attempted
- **Reduced step length**: 0.08m → 0.04m → 0.03m ✓ (helped, but still fails)
- **Reduced swing height**: 0.05m → 0.02m ✓ (helped, but still fails)
- **Increased IK iterations**: 5 → 12 ✓ (slight improvement)
- **Added weight-shift timing**: Smooth ramp over 35% of duration ✓ (slight improvement)
- **Removed hard support-margin cutoff**: Allows step to progress further, better visibility ✓

## Debugging Recommendations

### Step 1: Analyze Hip Roll Joint
The right hip roll joint is the first to fail (torque -21 Nm at collapse). 
- Check if this is a mechanical limit issue (run `python -c "from config import cfg; print(cfg.ROBOT.MAX_TORQUE)"`)
- Consider: Is 150 Nm a realistic G1 limit? Check robot datasheet.
- Solution: Reduce lateral CoM displacement, reduce weight-shift speed, or increase hip roll IK weight

### Step 2: Track CoM Lateral Drift
Print full CoM trajectory and compare to target:
- At tick 0: CoM target Y ≈ -0.04 m, actual Y ≈ 0.0 m (good)
- At tick 50: CoM target Y ≈ -0.04 m, actual Y ≈ -0.058 m (drifting)
- At tick 100: CoM target Y ≈ -0.04 m, actual Y ≈ -0.254 m (severely drifting)

Solution: Check if IK is tracking CoM target properly; consider increasing `KP_COM` from 60 to 100+.

### Step 3: Validate Swing Foot Tracking
The left foot swing target is set, but actual position lags:
- Compare `lf_act` vs `lf_tgt` at each checkpoint
- If gap grows linearly, the servo is not strong enough
- If gap appears suddenly, the IK solver may be failing

Solution: Reduce swing height further (< 0.01 m), or increase `KP_FOOT_POSITION` from 200 to 300+.

### Step 4: Check IK Convergence
Print IK task errors (foot position, foot orientation, CoM) during the step:
```python
# In phase_manager.py, after IK solve
print(f"  IK errors: foot_pos={err_lf_p:.4f}, foot_rot={err_lf_r:.4f}, com={err_com:.4f}")
```
If any error remains large after solving, the IK may be hitting joint limits.

### Step 5: Consider Alternative Control Strategies
- **Reduced preview horizon**: Shorter CoM trajectory window (current: 1.6s)
- **Force-based balance**: Use foot contact forces instead of CoM targets
- **Shorter step window**: Further reduce duration (current: 1.5s) to minimize drift time
- **Multi-step sequence**: Try 2–3 small steps instead of one larger step

## Configuration Guide

### Tuning Parameters (in `config.py`)

**Step Gait**:
```python
class ZMPControllerConfig:
    STEP_LENGTH = 0.03      # Reduce for stability
    SWING_HEIGHT = 0.02     # Increase for foot clearance
    STEP_TIME = 0.6         # Total step duration (not currently used)
```

**IK Gains**:
```python
class WholeBodyIKConfig:
    KP_FOOT_POSITION = 200  # Increase for better foot tracking (200–300)
    KP_COM = 60             # Increase for better CoM stability (60–100)
    TRACKING_ITERATIONS = 5 # Iterations per control step (5–12)
```

**Fall Detection (in Phase 5)**:
```python
# In StatusMonitor.is_fallen():
com_height < 0.65       # CoM height threshold during step
base_height < 0.68      # Floating base height threshold
tilt > 0.60             # Base rotation magnitude (quaternion-derived)
```

## File Structure

```
ZMP_Humanoid/
├── README.md                              # This file
├── scripts/
│   ├── main.py                           # Entry point
│   ├── phase_manager.py                  # Phase orchestration & control loop
│   ├── whole_body_ik.py                  # IK solver
│   ├── g1_robot_model.py                 # Robot kinematics
│   ├── joint_controller.py               # Actuator interface
│   ├── zmp_controller.py                 # ZMP preview control (Phase 4)
│   ├── config.py                         # Configuration parameters
│   ├── robot_model.py                    # Base robot class
│   └── zmp_*.py                          # ZMP analysis utilities
├── assets/
│   └── robot_descriptions/
│       ├── mujoco_menagerie/             # MuJoCo model collection
│       │   └── unitree_g1/               # G1 robot model
│       │       ├── g1.xml                # Main model file
│       │       ├── scene.xml             # Scene with gravity
│       │       └── assets/               # URDF, textures, etc.
│       └── unitree_ros/                  # ROS files (reference)
└── content_capsules/
    └── context_capsule_zmp_project.md    # Project context notes
```

## References

- **Robot**: Unitree G1 Humanoid (https://www.unitreerobotics.com/)
- **Simulator**: MuJoCo (https://www.deepmind.com/mujoco)
- **Control**: ZMP (Zero Moment Point) preview control
- **IK**: Damped weighted least-squares (DWLS) method

## Quick Start Troubleshooting

| Problem | Likely Cause | Quick Fix |
|---------|--------------|-----------|
| Robot falls immediately | Initial settling failed | Increase Phase 1 duration in `main.py` |
| Phases 1-4 work, Phase 5 fails | Step too aggressive | Reduce `STEP_LENGTH` or `SWING_HEIGHT` in `config.py` |
| No output from `main.py` | Missing MuJoCo or assets | Run `pip install mujoco` and verify `assets/` folder exists |
| Viewer window won't open | Display/graphics issue | Try running on a machine with GPU or use headless mode |
| IK solver not converging | Poor initialization | Check that Phase 2 completed successfully (CoM at foot center) |

## Contact & Support

For questions or issues:
1. Check the diagnostics output format (see "Diagnostics Output" section)
2. Review the Phase 5 failure modes (see "Known Issues")
3. Run the debugging recommendations (see "Debugging Recommendations")
4. Review Phase 5 joint diagnostics to identify which joint(s) are saturating

---

**Last Updated**: After Phase 5 instrumentation and fall detection improvements  
**Current Focus**: Stabilizing Phase 5 single-step gait via hip roll joint analysis and CoM lateral drift reduction  
**Next Priority**: Reduce weight-shift aggressiveness or increase IK foot-position tracking gains
