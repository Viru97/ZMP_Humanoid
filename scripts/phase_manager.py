import mujoco
import numpy as np
import time
from typing import Optional
# from robot_model import G1RobotModel
from g1_robot_model import G1RobotModel
from zmp_controller import ZMPPreviewController
from whole_body_ik import WholeBodyIK
from joint_controller import JointController
from dataclasses import dataclass
from config import cfg

# ================================================================
# VERBOSE DEBUG LOGGING
# ================================================================
DBG = True
DBG_TICK_INTERVAL = 10


def _dbg(*args, **kwargs):
    if DBG:
        print("  [DBG]", *args, **kwargs)


# ================================================================
# MODULE 5: Status Display
# ================================================================
@dataclass
class RobotStatus:
    """Snapshot of robot state for monitoring and logging."""
    com: np.ndarray           # Center of mass position [x,y,z] meters
    foot_center: np.ndarray   # Midpoint of feet [x,y,z] meters
    com_height: float         # CoM z-coordinate [meters]
    xy_error: float           # Horizontal distance: CoM to foot center [meters]
    phase: str                # Current control phase name
    tick: int                 # Current tick counter
    extra: str = ""           # Additional diagnostic info


class StatusMonitor:
    def __init__(self, robot: G1RobotModel):
        self.robot = robot
        self.history = []
        self.fall_threshold = cfg.ROBOT.FALL_HEIGHT_THRESHOLD

    def snapshot(self, data: mujoco.MjData, phase: str, tick: int,
                 target: Optional[np.ndarray] = None) -> RobotStatus:
        mujoco.mj_kinematics(self.robot.model, data)
        com = self.robot.get_com(data)
        fc = self.robot.get_foot_center(data)
        # XY error: horizontal distance from CoM projection to foot center
        # This indicates balance quality — should be < 0.02m for stable standing
        xy_err = np.linalg.norm(com[:2] - fc[:2])

        extra = ""
        if target is not None:
            # Tracking error: how far actual CoM is from commanded target
            tgt_err = np.linalg.norm(com[:2] - target[:2])
            extra = f"tgt_err={tgt_err:.4f}"

        status = RobotStatus(
            com=com, foot_center=fc, com_height=com[2],
            xy_error=xy_err, phase=phase, tick=tick, extra=extra
        )
        self.history.append(status)
        return status

    def print_status(self, status: RobotStatus):
        """Pretty-print a status snapshot."""
        print(f"  [{status.phase:8s}] tick={status.tick:5d} | "
              f"h={status.com_height:.4f} | "
              f"CoM=[{status.com[0]:.3f},{status.com[1]:.3f},{status.com[2]:.3f}] | "
              f"FC=[{status.foot_center[0]:.3f},{status.foot_center[1]:.3f}] | "
              f"xy_err={status.xy_error:.4f} {status.extra}")

    def base_tilt(self, data: mujoco.MjData) -> float:
        """Return the free-base tilt angle in radians relative to upright."""
        base_quat = data.qpos[3:7].copy()
        rot_flat = np.zeros(9)
        mujoco.mju_quat2Mat(rot_flat, base_quat)
        rot = rot_flat.reshape(3, 3)
        up_axis = rot[:, 2]
        return float(np.arccos(np.clip(up_axis[2], -1.0, 1.0)))

    def is_fallen(self, data: mujoco.MjData, threshold: float = None) -> bool:
        """
        Check if robot has fallen.

        BUG FIX: The previous version hardcoded 0.68m thresholds and ignored
        the `threshold` parameter entirely, causing spurious fall detection
        during normal stepping motion (natural CoM height ~0.695m, tiny
        perturbations triggered the 0.68m cutoff). Also the tilt limit of
        0.60 rad (~34 deg) was too tight for the weight-shift phase.

        Now uses the threshold parameter (defaulting to config value), and
        only checks CoM height + tilt (not raw base height which is noisier).
        Tilt limit increased to 0.80 rad (~46 deg) to allow for step lean.
        """
        if threshold is None:
            threshold = self.fall_threshold
        com = self.robot.get_com(data)
        tilt = self.base_tilt(data)
        # FIX: use the threshold parameter; removed hardcoded 0.68 checks
        return com[2] < threshold or tilt > 0.80

    def support_margin(self, data: mujoco.MjData) -> float:
        """
        Approximate margin of the projected CoM to the foot support rectangle edges.

        Returns the minimum distance (meters) from the CoM projection to the
        nearest foot-edge along the sagittal (X) and lateral (Y) axes. Positive
        means safely inside the support region; negative means outside.
        """
        mujoco.mj_kinematics(self.robot.model, data)
        com = self.robot.get_com(data)
        # Foot contact positions
        lf = data.xpos[self.robot.left_foot_id][:2].copy()
        rf = data.xpos[self.robot.right_foot_id][:2].copy()
        fc = (lf + rf) / 2.0

        dx = com[0] - fc[0]
        dy = com[1] - fc[1]

        half_x = cfg.ZMP.FOOT_LENGTH / 2.0
        half_y = cfg.ZMP.FOOT_WIDTH / 2.0

        margin_x = half_x - abs(dx)
        margin_y = half_y - abs(dy)

        return min(margin_x, margin_y)

    def is_margin_critical(self, data: mujoco.MjData, thresh: float = None) -> bool:
        """
        Return True if the computed support margin is below configured safety thresholds.
        """
        if thresh is None:
            # Use conservative threshold from phase config
            thresh = min(cfg.PHASE.COM_X_MARGIN, cfg.PHASE.COM_Y_MARGIN)
        return self.support_margin(data) < thresh


# ================================================================
# MODULE 6: Visualized Phase Manager
# ================================================================
class VisualizedPhaseManager:
    """
    Orchestrates all control phases with live visualization.

    Phases:
    1. Settle: Hold default pose, let physics find equilibrium
    2. Balance: Shift CoM over support polygon center
    3. Stability Hold: Verify robot stays balanced without falling
    4. ZMP Sway: Run preview controller with lateral oscillation

    The viewer is active throughout all phases so you can watch
    the robot's behavior and identify exactly where problems occur.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 viewer: mujoco.viewer.Handle):
        """
        Parameters:
            model: MuJoCo model (physics parameters, geometry, etc.)
            data: MuJoCo data (simulation state: positions, velocities, forces)
            viewer: Active viewer handle (must already be launched)
        """
        self.model = model
        self.data = data
        self.viewer = viewer

        # Create sub-modules
        # Instantiate the runtime robot model wrapper using the active MuJoCo model
        self.robot = G1RobotModel(model)
        self.controller = JointController(self.robot)
        self.ik = WholeBodyIK(self.robot)
        self.monitor = StatusMonitor(self.robot)

        # --- Timing parameters from config ---
        # sim_dt: Physics simulation timestep [seconds]
        #   Set in model XML. Smaller → more accurate physics but slower.
        #   0.001s (1kHz) is standard for contact-rich humanoid simulation.
        self.sim_dt = model.opt.timestep

        # ctrl_dt: Control loop period [seconds]
        #   How often we run IK + update control targets.
        #   0.01s (100Hz) is typical for whole-body control.
        #   Must be ≥ sim_dt. The gap is filled by sub-stepping.
        self.ctrl_dt = cfg.SIMULATION.CONTROL_DT

        # steps_per_ctrl: Number of physics steps per control update
        #   = ctrl_dt / sim_dt = 0.01 / 0.001 = 10
        #   Between each IK solve, the simulation runs 10 physics steps
        #   with the SAME control target held constant.
        self.steps_per_ctrl = max(1, int(self.ctrl_dt / self.sim_dt))

        # Get phase configuration
        self.phase_cfg = cfg.PHASE

        # Get robot configuration for thresholds
        self.robot_cfg = cfg.ROBOT

        # Get controller configuration
        self.ctrl_cfg = cfg.CONTROLLER

        print(f"  [Sim] sim_dt={self.sim_dt:.4f}s, ctrl_dt={self.ctrl_dt:.3f}s, "
              f"sub-steps={self.steps_per_ctrl}")

        # Resolve mocap body indices for visualization markers
        marker_names = {
            "actual_com": "marker_actual_com",
            "target_com": "marker_target_com",
            "ref_zmp":    "marker_ref_zmp",
            "lf":         "marker_lf",
            "rf":         "marker_rf",
        }
        self._mocap_idx: dict = {}
        for key, name in marker_names.items():
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id < 0:
                self._mocap_idx[key] = -1
            else:
                mocap_id = model.body_mocapid[body_id]
                self._mocap_idx[key] = mocap_id

    def _update_markers(self, actual_com: np.ndarray, target_com: np.ndarray,
                        ref_zmp_xy: np.ndarray, lf_pos: np.ndarray,
                        rf_pos: np.ndarray):
        """
        Write current positions to the 5 mocap visualization markers.

        Parameters:
            actual_com:  shape (3,) — current whole-body CoM
            target_com:  shape (3,) — IK CoM target
            ref_zmp_xy:  shape (2,) — reference ZMP in XY plane
            lf_pos:      shape (3,) — left foot world position
            rf_pos:      shape (3,) — right foot world position
        """
        n_mocap = self.data.mocap_pos.shape[0]

        _MARKER_Z = 0.005  # Raise floor-level markers just above ground [m]

        def _set(key: str, pos: np.ndarray):
            idx = self._mocap_idx.get(key, -1)
            if 0 <= idx < n_mocap:
                self.data.mocap_pos[idx] = pos

        _set("actual_com", actual_com)
        _set("target_com", target_com)
        _set("ref_zmp", np.array([ref_zmp_xy[0], ref_zmp_xy[1], _MARKER_Z]))
        _set("lf",      np.array([lf_pos[0],     lf_pos[1],     _MARKER_Z]))
        _set("rf",      np.array([rf_pos[0],     rf_pos[1],     _MARKER_Z]))

    def step_sim(self, q_target: np.ndarray, n_sub: Optional[int] = None,
                 sync_viewer: bool = True, realtime: bool = True,
                 com_target: Optional[np.ndarray] = None,
                 ref_zmp_xy: Optional[np.ndarray] = None):
        """
        Execute one control tick: apply control + sub-step physics + sync viewer.

        Parameters:
            q_target:    np.ndarray [nq] - desired joint configuration
            n_sub:       int or None - number of physics sub-steps (default: steps_per_ctrl)
            sync_viewer: bool - whether to update the viewer after stepping
            realtime:    bool - (currently unused, pacing is done externally)
            com_target:  Optional[np.ndarray] shape (3,) — IK CoM target for marker;
                         defaults to actual CoM if None
            ref_zmp_xy:  Optional[np.ndarray] shape (2,) — reference ZMP for marker;
                         defaults to actual CoM XY if None
        """
        n = n_sub if n_sub else self.steps_per_ctrl
        for _ in range(n):
            # Apply control signal to actuators
            self.controller.set_targets_from_qpos(self.data, q_target)
            # Advance physics by one sim_dt step
            mujoco.mj_step(self.model, self.data)

        # Update visualization markers before syncing the viewer
        actual_com = self.robot.get_com(self.data)
        _target_com = com_target if com_target is not None else actual_com
        _ref_zmp_xy = ref_zmp_xy if ref_zmp_xy is not None else actual_com[:2]
        lf_pos = self.data.xpos[self.robot.left_foot_id]
        rf_pos = self.data.xpos[self.robot.right_foot_id]
        self._update_markers(actual_com, _target_com, _ref_zmp_xy, lf_pos, rf_pos)

        # Update viewer display
        if sync_viewer and self.viewer.is_running():
            self.viewer.sync()

    def _print_step_joint_diagnostics(self, q_target: np.ndarray, limit: int = 8,
                                       show_all: bool = False):
        """
        Print actuator-space tracking diagnostics for the current step.

        The G1 uses position-controlled actuators, so the meaningful "torque"
        signal is the PD estimate from the commanded position error and joint
        velocity.
        """
        rows = []
        for act_id in range(self.robot.nu):
            qpos_idx = self.robot.act_to_qpos[act_id]
            dof_idx = self.robot.act_to_dof[act_id]
            jnt_id = self.robot.act_to_jnt[act_id]
            if qpos_idx < 0 or dof_idx < 0 or jnt_id < 0:
                continue

            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id) or f"act_{act_id}"
            q_current = float(self.data.qpos[qpos_idx])
            q_desired = float(q_target[qpos_idx])
            q_error = q_desired - q_current
            q_velocity = float(self.data.qvel[dof_idx])

            if self.robot.is_position_controlled:
                kp = float(self.model.actuator_gainprm[act_id, 0])
                kd = float(-self.model.actuator_biasprm[act_id, 2])
                tau_est = kp * q_error - kd * q_velocity
                ctrl_cmd = q_desired
                tau_act = float(self.data.actuator_force[act_id])
            else:
                ctrl_cmd = float(self.data.ctrl[act_id])
                tau_est = ctrl_cmd
                tau_act = float(self.data.actuator_force[act_id])

            rows.append((abs(q_error), joint_name, q_current, q_desired, q_error, q_velocity, ctrl_cmd, tau_est, tau_act))

        if not rows:
            return

        rows.sort(key=lambda row: row[0], reverse=True)
        selected = rows if show_all else rows[:limit]
        print("    [STEP JOINTS] name                 q_cur     q_tgt     q_err     qd       cmd      tau_act")
        for _, joint_name, q_current, q_desired, q_error, q_velocity, ctrl_cmd, tau_est, tau_act in selected:
            print(
                f"    [STEP JOINTS] {joint_name:20s} "
                f"{q_current:+.4f}  {q_desired:+.4f}  {q_error:+.4f}  "
                f"{q_velocity:+.4f}  {ctrl_cmd:+.4f}  {tau_act:+.2f}"
            )

        max_err = rows[0][0]
        rms_err = float(np.sqrt(np.mean([row[4] ** 2 for row in rows])))
        max_tau = max(abs(row[8]) for row in rows)
        print(f"    [STEP JOINTS] max_err={max_err:.4f} rad | rms_err={rms_err:.4f} rad | max|tau|={max_tau:.2f}")

    # ============================
    # PHASE 1: Settle
    # ============================
    def phase_settle(self, duration: float = None) -> bool:
        """
        Let robot settle into natural standing under default pose control.

        No IK, no CoM tracking — just hold joint defaults and let gravity
        pull the robot into a stable configuration. This establishes the
        "natural" standing height and foot positions.

        Parameters:
            duration: float [seconds] - how long to settle
                     Default from config. Enough for transients to die out.
                     If robot is still bouncing after this, there's a model issue.

        Returns:
            bool: True if robot is still standing (CoM > threshold), False if fallen.
        """
        if duration is None:
            duration = self.phase_cfg.SETTLE_TIME

        print("\n" + "="*60)
        print("  PHASE 1: SETTLING (default pose hold)")
        print("="*60)

        # Total ticks at control rate
        n_ticks = int(duration / self.ctrl_dt)
        # Target: model's default joint angles
        q_default = self.model.qpos0.copy()

        # Status reporting interval in ticks
        status_interval = int(self.phase_cfg.STATUS_INTERVAL / self.ctrl_dt)

        for tick in range(n_ticks):
            self.step_sim(q_default)

            # Report at configured interval
            if tick % status_interval == 0:
                status = self.monitor.snapshot(self.data, "SETTLE", tick)
                self.monitor.print_status(status)

                # threshold=0.25: very low — only catch complete collapses
                if self.monitor.is_fallen(self.data, threshold=0.25):
                    print("  *** FALLEN during settle! ***")
                    return False

            if not self.viewer.is_running():
                return False

            # Sleep for visual pacing (configurable multiplier for real-time)
            time.sleep(self.ctrl_dt * self.phase_cfg.SETTLE_PACE)

        com = self.robot.get_com(self.data)
        print(f"  SETTLE COMPLETE: CoM height = {com[2]:.4f}m")
        # Use configured minimum standing height threshold
        return com[2] > self.robot_cfg.MIN_STANDING_HEIGHT

    # ============================
    # PHASE 2: Balance acquisition
    # ============================
    def phase_balance(self, duration: float = None) -> bool:
        """
        Gradually shift CoM horizontally over support polygon center.

        The robot starts with CoM possibly offset from foot center (due to
        asymmetric mass distribution or settling). This phase slowly moves
        CoM XY to be directly above the midpoint of the feet.

        Uses IK in "tracking mode": syncs with sim every tick to stay close
        to reality. This prevents the IK planner from diverging from the
        actual robot state.

        Parameters:
            duration: float [seconds] - total time for balance acquisition
                     Default from config. Gives a gentle, stable ramp.

        Returns:
            bool: True if balanced (height > threshold), False if fallen.
        """
        if duration is None:
            duration = self.phase_cfg.BALANCE_TIME

        print("\n" + "="*60)
        print("  PHASE 2: BALANCE ACQUISITION (CoM -> foot center)")
        print("="*60)

        # Initialize IK from current simulation state
        self.ik.sync_from_sim(self.data.qpos)
        # Lock current foot positions as IK constraints
        # From this point, IK will keep feet at these exact poses.
        self.ik.capture_foot_targets()

        n_ticks = int(duration / self.ctrl_dt)
        # ramp_ticks: configurable fraction of duration for the ramp, rest for holding at target
        # This ensures smooth approach + time to verify stability at the end.
        ramp_ticks = int(n_ticks * self.phase_cfg.BALANCE_RAMP_RATIO)

        initial_com = self.robot.get_com(self.data)
        mujoco.mj_kinematics(self.model, self.data)
        foot_center = self.robot.get_foot_center(self.data)

        print(f"  Starting CoM: [{initial_com[0]:.4f}, {initial_com[1]:.4f}, {initial_com[2]:.4f}]")
        print(f"  Foot center:  [{foot_center[0]:.4f}, {foot_center[1]:.4f}, {foot_center[2]:.4f}]")

        # Status reporting interval in ticks
        status_interval = int(self.phase_cfg.STATUS_INTERVAL / self.ctrl_dt)

        # Get IK iteration count for tracking mode
        ik_tracking_iter = cfg.IK.TRACKING_ITERATIONS
        sync_each_tick = not self.robot.is_position_controlled

        for tick in range(n_ticks):
            # --- Compute smooth interpolation factor ---
            # alpha: 0.0 at start → 1.0 when fully ramped
            # Linear first, then apply smoothstep for zero velocity at endpoints
            alpha = min(1.0, tick / max(ramp_ticks, 1))
            # Smoothstep: f(x) = 3x² - 2x³
            # Has zero derivative at x=0 and x=1, so no sudden velocity jumps.
            # This prevents jerky motion that could destabilize the robot.
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)

            # Get current state
            mujoco.mj_kinematics(self.model, self.data)
            fc = self.robot.get_foot_center(self.data)
            com = self.robot.get_com(self.data)

            # --- CoM target: blend from initial toward foot center ---
            # XY: interpolate from the phase-start CoM to foot center.
            # Using the current CoM here causes the target to chase drift.
            # Z: DON'T control — let physics determine height naturally.
            #    Trying to force a specific height fights gravity and
            #    can cause instability (this was a key bug in original code).
            com_target = np.array([
                (1.0 - alpha) * initial_com[0] + alpha * foot_center[0],
                (1.0 - alpha) * initial_com[1] + alpha * foot_center[1],
                com[2]
            ])

            # For torque control we can safely re-sync every tick.
            # For position servos, per-tick sync collapses command error
            # (q_target ~= q_current), which removes support torque.
            if sync_each_tick:
                self.ik.sync_from_sim(self.data.qpos)

            # Solve IK in planner space and command the resulting joint target.
            q_target = self.ik.solve(com_target, dt=self.ctrl_dt, n_iter=ik_tracking_iter)
            self.step_sim(q_target, com_target=com_target)

            # Report at configured interval
            if tick % status_interval == 0:
                status = self.monitor.snapshot(self.data, "BALANCE", tick, com_target)
                self.monitor.print_status(status)
                if self.monitor.is_fallen(self.data):
                    print("  *** FALLEN during balance! ***")
                    return False

            if not self.viewer.is_running():
                return False

            # Configurable real-time pacing
            time.sleep(self.ctrl_dt * self.phase_cfg.BALANCE_PACE)

        com = self.robot.get_com(self.data)
        fc = self.robot.get_foot_center(self.data)
        print(f"  BALANCE COMPLETE: h={com[2]:.4f}, xy_err={np.linalg.norm(com[:2]-fc[:2]):.4f}")
        return com[2] > self.robot_cfg.MIN_STANDING_HEIGHT

    # ============================
    # PHASE 3: Stability hold
    # ============================
    def phase_stability_hold(self, duration: float = None) -> bool:
        """
        Hold the balanced position and verify the robot doesn't drift or fall.

        This is a sanity check before starting ZMP control.
        If the robot can't hold still for the configured time, ZMP sway will definitely fail.

        Parameters:
            duration: float [seconds] - hold duration
                     Default from config. Enough to detect slow drift or oscillation buildup.

        Returns:
            bool: True if stable throughout, False if fallen.
        """
        if duration is None:
            duration = self.phase_cfg.STABILITY_HOLD_TIME

        print("\n" + "="*60)
        print("  PHASE 3: STABILITY HOLD")
        print("="*60)

        # Re-sync IK and re-capture foot targets
        # This ensures we're holding the ACTUAL current configuration,
        # not some old target that may have drifted.
        self.ik.sync_from_sim(self.data.qpos)
        self.ik.capture_foot_targets()

        mujoco.mj_kinematics(self.model, self.data)
        fc = self.robot.get_foot_center(self.data)
        com = self.robot.get_com(self.data)
        hold_target = np.array([fc[0], fc[1], com[2]])

        n_ticks = int(duration / self.ctrl_dt)
        max_drift = 0.0

        # Status reporting interval in ticks
        status_interval = int(self.phase_cfg.STATUS_INTERVAL / self.ctrl_dt)

        # Get IK iteration count for tracking mode
        ik_tracking_iter = cfg.IK.TRACKING_ITERATIONS
        sync_each_tick = not self.robot.is_position_controlled

        for tick in range(n_ticks):
            # Keep periodic re-sync only in torque mode.
            if sync_each_tick:
                self.ik.sync_from_sim(self.data.qpos)
            q_target = self.ik.solve(hold_target, dt=self.ctrl_dt, n_iter=ik_tracking_iter)
            self.step_sim(q_target, com_target=hold_target)

            if tick % status_interval == 0:
                status = self.monitor.snapshot(self.data, "HOLD", tick, hold_target)
                self.monitor.print_status(status)
                max_drift = max(max_drift, status.xy_error)

                if self.monitor.is_fallen(self.data):
                    print("  *** FALLEN during hold! ***")
                    return False

            if not self.viewer.is_running():
                return False

            time.sleep(self.ctrl_dt * self.phase_cfg.HOLD_PACE)

        print(f"  HOLD COMPLETE: max XY drift = {max_drift:.4f}m")
        return True

    # ============================
    # PHASE 4: ZMP Preview Sway
    # ============================
    def phase_zmp_sway(self, duration: float = None,
                       amplitude: float = None,
                       frequency: float = None) -> bool:
        """
        Run ZMP preview control with lateral (Y-axis) sway.

        The preview controller generates a smooth CoM trajectory that
        keeps the ZMP tracking a sinusoidal reference. The robot sways
        side-to-side while maintaining balance.

        Parameters:
            duration: float [seconds] - total sway time
                     Default from config. Gives multiple full sway cycles.

            amplitude: float [meters] - peak lateral sway distance
                      Default from config. Conservative start — well within support polygon.
                      G1 foot width ~0.08m each, stance ~0.2m.
                      Support polygon half-width ~0.14m.
                      Default << 0.14m → very safe.
                      Can increase to 0.04-0.06m for more dramatic motion.

            frequency: float [Hz] - sway oscillation frequency
                      Default from config.
                      Natural pendulum frequency for h=0.7m: sqrt(g/h) / (2π) ≈ 0.6Hz
                      Default is well below resonance → smooth, easy to track.
                      Higher freq (0.5Hz) needs more aggressive control.

        Returns:
            bool: True if completed without falling, False otherwise.
        """
        # Get defaults from config
        zmp_cfg = cfg.ZMP
        if duration is None:
            duration = self.phase_cfg.ZMP_SWAY_DURATION
        if amplitude is None:
            amplitude = zmp_cfg.SWAY_AMPLITUDE
        if frequency is None:
            frequency = zmp_cfg.SWAY_FREQUENCY

        print("\n" + "="*60)
        print("  PHASE 4: ZMP PREVIEW CONTROL")
        print(
            f"  amplitude={amplitude*100:.1f}cm, freq={frequency:.2f}Hz, "
            f"duration={duration:.0f}s"
        )
        print(f"  mode={self.phase_cfg.PHASE4_MODE}")
        print("=" * 60)
        self.ik.sync_from_sim(self.data.qpos)
        self.ik.capture_foot_targets()

        mode = str(self.phase_cfg.PHASE4_MODE).strip().lower()
        use_safe_sway = (mode == "safe_sway")
        if mode not in ("safe_sway", "preview_ik"):
            print(f"  [ZMP] Unknown mode '{self.phase_cfg.PHASE4_MODE}', falling back to safe_sway.")
            use_safe_sway = True

        if use_safe_sway:
            print("  [ZMP] Using safe_sway fallback (waist-roll sway).")

            def _find_joint_qpos_idx(name_tokens):
                for j in range(self.model.njnt):
                    jname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
                    jname_l = jname.lower()
                    if any(tok in jname_l for tok in name_tokens):
                        return self.model.jnt_qposadr[j]
                return None

            waist_roll_idx = _find_joint_qpos_idx(["waist_roll_joint", "waist_roll"])
            if waist_roll_idx is None:
                print("  [ZMP] waist_roll joint not found, skipping sway to stay safe.")
                return True

            n_ticks = int(duration / self.ctrl_dt)
            status_interval = max(1, int(self.phase_cfg.STATUS_INTERVAL / self.ctrl_dt))
            # Use model default standing pose as the base posture for sway.
            # Using the current drifted pose can compound instability over time.
            base_q = self.model.qpos0.copy()

            # Convert requested lateral sway (meters) to a small waist-roll angle.
            # 4 rad/m maps 2cm -> 0.08rad (~4.6 deg), conservative for stability.
            roll_gain = 4.0
            max_roll = 0.10  # rad

            for tick in range(n_ticks):
                t = tick * self.ctrl_dt

                ramp = 0.0
                if t > zmp_cfg.SWAY_RAMP_START:
                    ramp = min(1.0, (t - zmp_cfg.SWAY_RAMP_START) / max(zmp_cfg.SWAY_RAMP_DURATION, 1e-6))
                    ramp = ramp * ramp * (3.0 - 2.0 * ramp)

                sway_y = amplitude * ramp * np.sin(2.0 * np.pi * frequency * max(0.0, t - zmp_cfg.SWAY_RAMP_START))
                roll_cmd = np.clip(roll_gain * sway_y, -max_roll, max_roll)

                q_target = base_q.copy()
                q_target[waist_roll_idx] = base_q[waist_roll_idx] + roll_cmd

                com_now = self.robot.get_com(self.data)
                com_target = np.array([com_now[0], com_now[1] + sway_y, com_now[2]])
                self.step_sim(q_target, com_target=com_target,
                              ref_zmp_xy=np.array([com_now[0], com_now[1] + sway_y]))

                if tick % status_interval == 0:
                    status = self.monitor.snapshot(self.data, "ZMP", tick, com_target)
                    self.monitor.print_status(status)
                    if self.monitor.is_fallen(self.data):
                        print("  *** FALLEN during ZMP control! ***")
                        return False

                if not self.viewer.is_running():
                    return False

                time.sleep(self.ctrl_dt * self.phase_cfg.SWAY_PACE)

        print("\n  ZMP SWAY COMPLETE.")
        return True

    # ============================
    # PHASE 5: Single Step Walking
    # ============================
    def phase_single_step(self, step_forward: float = 0.03, duration: float = 2.5) -> bool:
        """
        Execute a single forward step: right foot support, left foot swings forward.

        BUG FIXES applied vs original:
        ------------------------------------------------------------------
        FIX 1 — support_com_target Y was wrong.
            Old: right_foot_pos[1] - 0.04  → puts CoM at the OUTER EDGE of
                 the right foot (4cm beyond foot center toward negative Y).
                 Confirmation: right_foot_pos[1] ≈ -0.119,
                 target = -0.159 = exactly outer edge of 8cm-wide foot.
                 This requires shifting CoM 16.2cm at 23cm/s → robot topples.
            Fix: right_foot_pos[1]          → targets foot CENTER (stable).

        FIX 2 — Weight-shift fraction was 35%, leaving only 0.52s at duration=1.5s
            to shift CoM 12cm. That's 23cm/s — far too aggressive.
            Fix: increased to 55%, and default duration → 2.5s.
            Result: 1.38s to shift 12cm = 8.8cm/s (safe).

        FIX 3 — ik_sync_interval was 5 ticks (50ms).
            Frequent sync copies the drifting sim base into the IK planner,
            causing the IK to compute corrections from an already-compromised
            state, which amplifies instability.
            Fix: sync every 15 ticks (150ms), which gives the planner
            enough inertia to stabilize without diverging too far.

        FIX 4 — is_fallen() ignored the threshold parameter (hardcoded 0.68m).
            0.68m is only 1.5cm below the standing height of 0.695m —
            any slight perturbation from the step motion triggered it.
            Fix: is_fallen() now uses the threshold parameter. Step phase
            passes threshold=0.60m (clearly fallen vs. intentional lean).

        FIX 5 — X offset of -0.005m on support_com_target was unnecessary noise.
            Fix: removed, now uses right_foot_pos[0] directly.
        ------------------------------------------------------------------
        """
        print("\n" + "="*60)
        print(f"  PHASE 5: SINGLE STEP (forward={step_forward:.3f}m, duration={duration:.2f}s)")
        print("="*60)

        self.ik.sync_from_sim(self.data.qpos)
        mujoco.mj_kinematics(self.model, self.data)

        left_foot_id  = self.robot.left_foot_id
        right_foot_id = self.robot.right_foot_id

        # Use xpos (body origin) — must match IK's capture_targets() which uses xpos.
        # xipos (body CoM) differs by up to 2.6cm on ankle_roll_link and would
        # create a constant offset error in foot tracking throughout the step.
        left_foot_pos  = self.data.xpos[left_foot_id].copy()
        right_foot_pos = self.data.xpos[right_foot_id].copy()

        n_ticks = int(duration / self.ctrl_dt)
        status_interval = max(1, int(self.phase_cfg.STATUS_INTERVAL / self.ctrl_dt))

        # --- FIX 2: Increased weight-shift fraction from 0.35 → 0.55 ---
        t_weight_shift = 0.55   # 55% of total time for weight shift (was 0.35)
        t_swing        = 0.85   # 85% of total time: swing ends here (was 0.75)
        t_landing      = 1.00   # 100%: landing + stabilise

        n_weight_shift = int(n_ticks * t_weight_shift)
        n_swing_end    = int(n_ticks * t_swing)

        swing_height = cfg.ZMP.SWING_HEIGHT
        ground_z     = left_foot_pos[2]

        left_landing_pos      = left_foot_pos.copy()
        left_landing_pos[0]  += step_forward

        initial_com  = self.robot.get_com(self.data)
        mujoco.mj_kinematics(self.model, self.data)
        foot_center  = self.robot.get_foot_center(self.data)

        # --- FIX 1: target foot CENTER not outer edge ---
        # Old (wrong): right_foot_pos[1] - 0.04  (= outer edge, outside stable zone)
        # New (correct): right_foot_pos[1]         (= foot centre, always stable)
        # --- FIX 5: removed unnecessary -0.005 X offset ---
        support_com_target = np.array([
            right_foot_pos[0],          # FIX 5: was right_foot_pos[0] - 0.005
            right_foot_pos[1],          # FIX 1: was right_foot_pos[1] - 0.04
            initial_com[2]
        ])

        step_ik_iter = max(cfg.IK.TRACKING_ITERATIONS, 12)

        # Sync every 5 ticks (50ms).
        # 15 ticks (150ms) was too long: the planner base drifts up to 0.72 rad
        # from the sim base between syncs (15 × 12 iters × 1ms × 4 rad/s), which
        # makes foot-constraint Jacobians computed at the wrong base pose. The
        # resulting joint commands physically launch both feet off the ground.
        ik_sync_interval = 5

        print(f"  Initial COM  : {initial_com.round(4)}")
        print(f"  Left foot    : {left_foot_pos.round(4)}")
        print(f"  Right foot   : {right_foot_pos.round(4)}")
        print(f"  Landing pos  : {left_landing_pos.round(4)}")
        print(f"  CoM target Y : {support_com_target[1]:.4f}  "
              f"(foot center={right_foot_pos[1]:.4f}, separation={abs(right_foot_pos[1]-left_foot_pos[1])*100:.0f}cm)")

        # Set initial foot targets (both feet planted)
        self.ik.lf_pos_target = left_foot_pos.copy()
        self.ik.lf_mat_target = self.data.xmat[left_foot_id].reshape(3, 3).copy()
        self.ik.rf_pos_target = right_foot_pos.copy()
        self.ik.rf_mat_target = self.data.xmat[right_foot_id].reshape(3, 3).copy()

        for tick in range(n_ticks):
            t_norm = tick / max(n_ticks, 1)

            # ============================================================
            # Phase A: Weight shift (0 → t_weight_shift)
            # Both feet planted; CoM moves toward right foot
            # ============================================================
            if tick < n_weight_shift:
                ws_progress = tick / max(n_weight_shift, 1)
                ws_progress = ws_progress * ws_progress * (3.0 - 2.0 * ws_progress)

                # BUG FIX: interpolate from initial_com (not foot_center).
                # The old code used foot_center as the starting point, which
                # is at [0,0] while the actual CoM starts at [0.031, 0.003].
                # At tick=0 ws_progress=0 → com_target=[0,0] but actual CoM
                # is [0.031,0.003], so the IK immediately sees a 3cm X error
                # and tries to correct it, pushing the robot forward into a fall.
                # Starting from initial_com gives zero error at tick=0. ✓
                com_target = np.array([
                    (1.0 - ws_progress) * initial_com[0] + ws_progress * support_com_target[0],
                    (1.0 - ws_progress) * initial_com[1] + ws_progress * support_com_target[1],
                    initial_com[2]
                ])

                # Keep both feet pinned
                self.ik.lf_pos_target = left_foot_pos.copy()
                self.ik.lf_mat_target = self.data.xmat[left_foot_id].reshape(3, 3).copy()
                self.ik.rf_pos_target = right_foot_pos.copy()
                self.ik.rf_mat_target = self.data.xmat[right_foot_id].reshape(3, 3).copy()

            # ============================================================
            # Phase B: Swing + landing (t_weight_shift → end)
            # Left foot arcs forward; CoM stays over right foot
            # ============================================================
            else:
                swing_progress = (tick - n_weight_shift) / max(n_swing_end - n_weight_shift, 1)
                swing_progress = np.clip(swing_progress, 0.0, 1.0)

                # Foot trajectory: sinusoidal arc
                left_foot_swing = left_foot_pos.copy()
                left_foot_swing[0] = left_foot_pos[0] + swing_progress * step_forward
                left_foot_swing[1] = left_foot_pos[1]

                if swing_progress < 1.0:
                    left_foot_swing[2] = ground_z + swing_height * np.sin(np.pi * swing_progress)
                else:
                    left_foot_swing[2] = ground_z

                self.ik.lf_pos_target = left_foot_swing.copy()
                self.ik.lf_mat_target = self.data.xmat[left_foot_id].reshape(3, 3).copy()

                com_target = support_com_target.copy()

                # Late swing (>80%): transition CoM toward new foot center
                if swing_progress > 0.8:
                    tp = (swing_progress - 0.8) / 0.2
                    tp = tp * tp * (3.0 - 2.0 * tp)
                    new_fc = (left_landing_pos + right_foot_pos) / 2.0
                    com_target = np.array([
                        (1.0 - tp) * support_com_target[0] + tp * new_fc[0],
                        (1.0 - tp) * support_com_target[1] + tp * new_fc[1],
                        initial_com[2]
                    ])

            # --- FIX 3: sync less frequently ---
            if tick % ik_sync_interval == 0:
                self.ik.sync_from_sim(self.data.qpos)

            q_target = self.ik.solve(com_target, dt=self.ctrl_dt, n_iter=step_ik_iter)
            self.step_sim(q_target, com_target=com_target)

            # ---- Diagnostics ----
            if tick % status_interval == 0:
                status = self.monitor.snapshot(self.data, "STEP", tick, com_target)
                self.monitor.print_status(status)
                self._print_step_joint_diagnostics(q_target, limit=8)

                lf_target = self.ik.lf_pos_target
                rf_target = self.ik.rf_pos_target
                lf_actual = self.data.xpos[left_foot_id]
                rf_actual = self.data.xpos[right_foot_id]
                print(f"    [STEP FEET] lf_act=[{lf_actual[0]:+.3f},{lf_actual[1]:+.3f},{lf_actual[2]:+.3f}] "
                      f"lf_tgt=[{lf_target[0]:+.3f},{lf_target[1]:+.3f},{lf_target[2]:+.3f}]")
                print(f"    [STEP FEET] rf_act=[{rf_actual[0]:+.3f},{rf_actual[1]:+.3f},{rf_actual[2]:+.3f}] "
                      f"rf_tgt=[{rf_target[0]:+.3f},{rf_target[1]:+.3f},{rf_target[2]:+.3f}]")

                # Support margin (single-support during swing)
                if tick < n_weight_shift:
                    margin = self.monitor.support_margin(self.data)
                else:
                    mujoco.mj_kinematics(self.model, self.data)
                    com_now = self.robot.get_com(self.data)
                    dx = com_now[0] - right_foot_pos[0]
                    dy = com_now[1] - right_foot_pos[1]
                    half_x = cfg.ZMP.FOOT_LENGTH / 2.0
                    half_y = cfg.ZMP.FOOT_WIDTH / 2.0
                    margin = min(half_x - abs(dx), half_y - abs(dy))

                print(f"    [STEP] support_margin={margin:.4f}m  "
                      f"phase={'weight-shift' if tick < n_weight_shift else 'swing'}")

                if margin <= -0.05:
                    print("  *** WARNING: support margin too negative ***")

                # threshold=0.60: catches the actual fall (h dropped to 0.579)
                # while still ignoring normal step lean (standing h ≈ 0.695).
                # 0.45 was too low — 0.579 > 0.45 passed undetected.
                if self.monitor.is_fallen(self.data, threshold=0.60):
                    print("  *** FALLEN during step! ***")
                    self._print_step_joint_diagnostics(q_target, limit=self.robot.nu, show_all=True)
                    return False

            if not self.viewer.is_running():
                return False

            time.sleep(self.ctrl_dt * self.phase_cfg.STEP_PACE)

        print(f"  SINGLE STEP COMPLETE.")
        return True