import mujoco
import mujoco.viewer
import time
import os
import sys
from typing import Tuple

# Import custom modules
#from robot_model import G1RobotModel
from phase_manager import VisualizedPhaseManager
from config import cfg

try:
    from robot_descriptions import g1_mj_description
except ImportError:
    print("Please install: pip install robot_descriptions")
    sys.exit(1)


# ================================================================
# MODULE 7: Model Loading
# ================================================================
def load_g1_model() -> Tuple[mujoco.MjModel, mujoco.MjData]:
    """
    Load G1 robot model with ground plane and tuned physics parameters.

    The scene wraps the robot model with:
    - A ground plane for contact
    - Tuned solver parameters for stable humanoid simulation
    - Appropriate friction and contact parameters

    Returns:
        (model, data): Tuple of MjModel and initialized MjData
    """
    print("\n" + "=" * 60)
    print("  LOADING G1 MODEL")
    print("=" * 60)

    xml_path = g1_mj_description.MJCF_PATH
    model_dir = os.path.dirname(xml_path)
    xml_file = os.path.basename(xml_path)

    # Get simulation parameters from config
    sim_cfg = cfg.SIMULATION
    gravity = sim_cfg.GRAVITY
    ground_friction = sim_cfg.GROUND_FRICTION
    ground_solref = sim_cfg.GROUND_SOLREF

    # Scene XML wraps the robot model and adds environment
    scene_xml = f"""
    <mujoco model="g1_zmp_scene">
        <!-- angle="radian": all angles in XML are in radians (not degrees) -->
        <compiler angle="radian"/>

        <!-- Physics solver options -->
        <option 
            timestep="{sim_cfg.DT}"
            iterations="{sim_cfg.SOLVER_ITERATIONS}"
            solver="{sim_cfg.SOLVER_TYPE}"
            tolerance="{sim_cfg.SOLVER_TOLERANCE}"
            gravity="{gravity[0]} {gravity[1]} {gravity[2]}"
            cone="{sim_cfg.CONE_TYPE}"
            impratio="{sim_cfg.IMPRATIO}"
        />
        <!--
        timestep: Physics step size [seconds]. 0.001s (1kHz) is standard for
                  contact-rich humanoid sim. Smaller → more accurate but slower.

        iterations: Max solver iterations per step. 50 is generous — ensures
                    convergence even for complex multi-contact scenarios.

        solver: "Newton" — Newton's method for constraint solving.
                More accurate than "PGS" (projected Gauss-Seidel) for humanoids.
                Converges faster for bilateral constraints (joints).

        tolerance: Solver convergence tolerance. 1e-10 ensures high accuracy.
                   Larger values (1e-6) are faster but may have constraint drift.

        gravity: [x, y, z] m/s². Standard Earth gravity pointing down.

        cone: "elliptic" — friction cone approximation.
              More accurate than "pyramidal" for circular foot contacts.
              Gives more realistic ground friction behavior.

        impratio: Impedance ratio for contacts. Higher (10) makes contacts
                  stiffer/harder. Prevents foot sinking into ground.
                  Range: 1 (soft) to 100 (very hard). 10 is good for rigid shoes.
        -->

        <include file="{xml_file}"/>

        <worldbody>
            <!-- Lighting for visualization -->
            <light pos="0 0 3" dir="0 0 -1" directional="true" 
                   diffuse="0.8 0.8 0.8" specular="0.3 0.3 0.3"/>
            <light pos="2 2 2" dir="-1 -1 -1" directional="false"
                   diffuse="0.4 0.4 0.4"/>

            <!-- Ground plane -->
            <geom name="ground" type="plane" size="10 10 0.1" 
                  rgba="0.75 0.85 0.75 1"
                  contype="1" conaffinity="1"
                  friction="{ground_friction[0]} {ground_friction[1]} {ground_friction[2]}"
                  solref="{ground_solref[0]} {ground_solref[1]}"
            />
            <!--
            type="plane": Infinite ground plane (size only affects rendering).

            contype/conaffinity: Contact filter bitmasks.
                contype=1, conaffinity=1: will collide with any geom that
                also has contype=1 or conaffinity=1. The robot's foot geoms
                should have matching values to enable ground contact.

            friction="mu_slide mu_spin mu_roll":
                mu_slide = 1.0: Sliding (Coulomb) friction coefficient.
                    1.0 means force needed to slide = 1.0 × normal force.
                    Rubber on concrete ≈ 0.7-1.0. Ensures feet don't slip.
                mu_spin = 0.005: Torsional (spinning) friction.
                    Prevents foot from spinning in place. Low value.
                mu_roll = 0.001: Rolling friction. Very low for flat contacts.

            solref="timeconst dampratio":
                timeconst = 0.004 [seconds]: Contact spring time constant.
                    Smaller → stiffer contact (less penetration).
                    0.004s at 0.001s timestep gives ~4:1 ratio (stable).
                dampratio = 1.0: Critical damping ratio for contact.
                    1.0 = critically damped (no bounce). 
                    <1 = underdamped (bouncy). >1 = overdamped (sluggish).
            -->
            <!-- VISUAL MARKERS -->
            <body name="marker_actual_com" mocap="true">
                <geom type="sphere" size="0.025" rgba="1 0 0 0.7" contype="0" conaffinity="0"/>
            </body>
            <body name="marker_target_com" mocap="true">
                <geom type="sphere" size="0.02" rgba="0 1 0 0.8" contype="0" conaffinity="0"/>
            </body>
            <body name="marker_ref_zmp" mocap="true">
                <geom type="sphere" size="0.025" rgba="0 0 1 0.8" contype="0" conaffinity="0"/>
            </body>

            <!-- Support Polygon Markers (Pads under feet) -->
            <body name="marker_lf" mocap="true">
                <geom type="box" size="0.08 0.04 0.005" rgba="1 1 0 0.4" contype="0" conaffinity="0"/>
            </body>
            <body name="marker_rf" mocap="true">
                <geom type="box" size="0.08 0.04 0.005" rgba="1 1 0 0.4" contype="0" conaffinity="0"/>
            </body>
        </worldbody>
    </mujoco>
    """

    # Must change directory so <include> finds the robot XML and meshes
    original_dir = os.getcwd()
    os.chdir(model_dir)
    try:
        model = mujoco.MjModel.from_xml_string(scene_xml)
    finally:
        os.chdir(original_dir)

    # Initialize simulation data from model
    data = mujoco.MjData(model)
    # Set initial joint angles to model default
    data.qpos[:] = model.qpos0[:]
    # Compute initial kinematics/dynamics
    mujoco.mj_forward(model, data)

    # Check initial height — model may start with base at origin (too low)
    mujoco.mj_comPos(model, data)
    h = data.subtree_com[0][2]
    print(f"  Model loaded: {xml_file}")
    print(f"  Timestep: {model.opt.timestep}s")
    print(f"  Initial CoM height: {h:.4f}m")

    if h < cfg.ROBOT.FALL_HEIGHT_THRESHOLD:
        # If model's default has robot at ground level, raise the floating base.
        # qpos[2] = base Z position for floating-base robots.
        # Initial height puts G1's feet approximately at ground level.
        print(f"  Raising base to {cfg.ROBOT.INITIAL_BASE_HEIGHT}m (model default too low)...")
        data.qpos[2] = cfg.ROBOT.INITIAL_BASE_HEIGHT
        mujoco.mj_forward(model, data)
        mujoco.mj_comPos(model, data)
        print(f"  Adjusted CoM height: {data.subtree_com[0][2]:.4f}m")

    return model, data


# ================================================================
# MODULE 8: Main Entry Point
# ================================================================
def main():
    """
    Main function: loads model, launches viewer IMMEDIATELY,
    then runs all control phases with full visualization.

    Execution flow:
    1. Load model and create simulation data
    2. Launch viewer (you see the robot from frame 0)
    3. Phase 1: Settle (3s) — robot finds natural standing under default control
    4. Phase 2: Balance (6s) — IK shifts CoM over feet
    5. Phase 3: Hold (3s) — verify stability before dynamic control
    6. Phase 4: ZMP sway (40s) — preview controller generates lateral motion

    Each phase has built-in fall detection. If the robot falls,
    the program stops with diagnostic messages visible in the viewer.
    """
    print("\n" + "#" * 60)
    print("#  ZMP PREVIEW CONTROL - UNITREE G1")
    print("#  Full visualization from start")
    print("#" * 60)

    # --- Load model ---
    model, data = load_g1_model()

    # *** LAUNCH VIEWER IMMEDIATELY ***
    # This is the KEY difference from the original code:
    # You can watch everything from the very first physics step.
    print("\n  Launching viewer (watch all phases)...")
    print("  Close viewer window to abort at any time.\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # --- Configure camera for good initial view ---
        vis_cfg = cfg.VISUALIZATION

        # distance: How far camera is from lookat point [meters]
        #   2.5m shows full robot with some surrounding context
        viewer.cam.distance = vis_cfg.CAMERA_DISTANCE

        # azimuth: Horizontal rotation around lookat [degrees]
        #   135° gives a 3/4 view (not pure front/side)
        viewer.cam.azimuth = vis_cfg.CAMERA_AZIMUTH

        # elevation: Vertical angle [degrees, negative = looking down]
        #   -20° looks slightly downward at the robot
        viewer.cam.elevation = vis_cfg.CAMERA_ELEVATION

        # lookat: Point the camera is centered on [x, y, z] meters
        #   [0, 0, 0.7] = roughly chest height of standing G1
        viewer.cam.lookat[:] = vis_cfg.CAMERA_LOOKAT

        # Sync viewer to show initial configuration before any stepping
        viewer.sync()

        # Brief pause so user can see the initial state
        time.sleep(vis_cfg.INITIAL_PAUSE)

        # Create phase manager (owns all control logic)
        manager = VisualizedPhaseManager(model, data, viewer)

        # === RUN PHASES SEQUENTIALLY ===
        phase_cfg = cfg.PHASE

        # Phase 1: Settle
        if not manager.phase_settle(duration=phase_cfg.SETTLE_TIME):
            print("\n*** PHASE 1 FAILED: Robot cannot stand with default control ***")
            print("    Possible fixes:")
            print("    - Check if model has proper actuator definitions")
            print("    - Increase initial qpos[2] (base height)")
            print("    - Check joint limits and default pose (qpos0)")
            print("    - Verify floor contact parameters match robot feet")
            _wait_for_viewer(viewer)
            return

        # Phase 2: Balance over feet
        if not manager.phase_balance(duration=phase_cfg.BALANCE_TIME):
            print("\n*** PHASE 2 FAILED: Could not achieve CoM over feet ***")
            print("    Possible fixes:")
            print("    - Reduce IK aggressiveness (lower kp_com in WholeBodyIK.solve)")
            print("    - Check foot body identification (see body list above)")
            print("    - Verify IK sync policy matches actuator type:")
            print("      position servos: avoid per-tick sync; torque mode: frequent sync")
            print("    - Try longer duration or smaller alpha ramp")
            _wait_for_viewer(viewer)
            return

        # Phase 3: Hold and verify
        if not manager.phase_stability_hold(duration=phase_cfg.STABILITY_HOLD_TIME):
            print("\n*** PHASE 3 FAILED: Not stable enough for ZMP control ***")
            print("    Possible fixes:")
            print("    - Increase phase 2 duration for better convergence")
            print("    - Check if foot constraints are holding (feet sliding?)")
            print("    - Verify PD gains are appropriate for this model")
            _wait_for_viewer(viewer)
            return

        # Phase 4: ZMP sway!
        print("\n  All stability phases passed! Starting ZMP sway...")
        zmp_cfg = cfg.ZMP
        manager.phase_zmp_sway(
            duration=phase_cfg.ZMP_SWAY_DURATION,
            amplitude=zmp_cfg.SWAY_AMPLITUDE,
            frequency=zmp_cfg.SWAY_FREQUENCY
        )

        # Phase 5: Single step walking
        print("\n  ZMP sway complete. Attempting single-step walking...")
        if manager.phase_single_step(step_forward=zmp_cfg.STEP_LENGTH, duration=2.5):
            print("\n*** SINGLE STEP SUCCESSFUL! ***")
        else:
            print("\n*** SINGLE STEP FAILED - robot could not complete step ***")

        # Keep viewer open after completion so user can inspect final state
        print("\n  Control complete. Viewer remains open.")
        _wait_for_viewer(viewer)


def _wait_for_viewer(viewer):
    """
    Keep viewer alive until user closes the window.
    Allows inspection of the robot's final state after control ends or fails.
    """
    print("  (Close viewer window to exit)")
    while viewer.is_running():
        viewer.sync()
        # Sleep time from config for idle viewer
        # Low enough to be responsive to user closing window
        time.sleep(cfg.VISUALIZATION.IDLE_SLEEP)


# ================================================================
if __name__ == "__main__":
    main()

