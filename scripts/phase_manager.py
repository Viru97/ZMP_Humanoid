import mujoco
import numpy as np
import time
from typing import Optional
from robot_model import G1RobotModel
from zmp_controller import ZMPPreviewController
from whole_body_ik import WholeBodyIK
from joint_controller import JointController
from dataclasses import dataclass
from config import cfg

# ================================================================
# VERBOSE DEBUG LOGGING
# Set DBG = True to enable per-tick diagnostic output.
# Set DBG = False once the issue is found to keep logs clean.
# ================================================================
DBG = True
DBG_TICK_INTERVAL = 10  # print every N control ticks when DBG=True


def _dbg(*args, **kwargs):
    """Print only when DBG is enabled."""
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
    """
    Prints and tracks robot status for debugging.
    Records history for post-hoc analysis.
    """

    def __init__(self, robot: G1RobotModel):
        self.robot = robot
        self.history = []  # List of RobotStatus snapshots
        # Get fall threshold from config
        self.fall_threshold = cfg.ROBOT.FALL_HEIGHT_THRESHOLD

    def snapshot(self, data: mujoco.MjData, phase: str, tick: int,
                 target: Optional[np.ndarray] = None) -> RobotStatus:
        """
        Take a status snapshot.

        Parameters:
            data: Current simulation state
            phase: Name of current phase (for display)
            tick: Current tick number
            target: Optional CoM target — if given, computes tracking error
        """
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

    def is_fallen(self, data: mujoco.MjData, threshold: float = None) -> bool:
        """
        Check if robot has fallen.

        Parameters:
            threshold: float [meters] - CoM height below which robot is "fallen"
                      Default from config.ROBOT.FALL_HEIGHT_THRESHOLD
                      0.30m is well below any humanoid's standing height.
                      G1 standing height is ~0.7m; anything below 0.30 is on the ground.
        """
        if threshold is None:
            threshold = self.fall_threshold
        com = self.robot.get_com(data)
        return com[2] < threshold


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

        # --- Resolve mocap body indices for visualization markers ---
        # Maps short key → mocap array index (or -1 if not found)
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

            # --- CoM target: blend from current toward foot center ---
            # XY: interpolate toward being directly above feet
            # Z: DON'T control — let physics determine height naturally.
            #    Trying to force a specific height fights gravity and
            #    can cause instability (this was a key bug in original code).
            com_target = np.array([
                (1.0 - alpha) * com[0] + alpha * fc[0],
                (1.0 - alpha) * com[1] + alpha * fc[1],
                com[2]  # Keep current height (let physics decide)
            ])

            # CRITICAL: Sync IK with sim every tick
            # "Tracking mode" — planner stays close to actual state.
            # Without this, small errors accumulate and the IK plans
            # configurations the robot can never achieve.
            self.ik.sync_from_sim(self.data.qpos)

            # Solve IK with tracking iterations (fast, for real-time tracking)
            # Fewer iterations because we sync every tick anyway.
            # The small residual error is corrected next tick.
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

        # Hold target: CoM directly above foot center, at current height
        mujoco.mj_kinematics(self.model, self.data)
        fc = self.robot.get_foot_center(self.data)
        com = self.robot.get_com(self.data)
        hold_target = np.array([fc[0], fc[1], com[2]])

        n_ticks = int(duration / self.ctrl_dt)
        max_drift = 0.0  # Track worst-case XY drift

        # Status reporting interval in ticks
        status_interval = int(self.phase_cfg.STATUS_INTERVAL / self.ctrl_dt)

        # Get IK iteration count for tracking mode
        ik_tracking_iter = cfg.IK.TRACKING_ITERATIONS

        for tick in range(n_ticks):
            # Track mode: sync every tick, solve, apply
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
        print(f"  amplitude={amplitude*100:.1f}cm, freq={frequency:.2f}Hz, "
              f"duration={duration:.0f}s")
        print("="*60)

        # Finalize IK from current state
        self.ik.sync_from_sim(self.data.qpos)
        self.ik.capture_foot_targets()

        # Get starting CoM position and height
        com = self.robot.get_com(self.data)
        # z_c: LIPM pendulum height — critical parameter for ZMP controller
        # Must match actual CoM height for the LIPM model to be accurate.
        z_c = com[2]

        # Create separate ZMP controllers for X and Y axes
        # X: sagittal (forward/backward) — held constant (no walking yet)
        # Y: lateral (side-to-side) — sway oscillation
        zmp_x = ZMPPreviewController(z_c=z_c, dt=self.ctrl_dt,
                                      preview_time=zmp_cfg.PREVIEW_TIME)
        zmp_y = ZMPPreviewController(z_c=z_c, dt=self.ctrl_dt,
                                      preview_time=zmp_cfg.PREVIEW_TIME)
        # Initialize both controllers at current CoM position
        zmp_x.reset(com[0])
        zmp_y.reset(com[1])

        # --- Generate ZMP reference trajectory ---
        n_ticks = int(duration / self.ctrl_dt)
        N_prev = zmp_x.N  # Preview horizon length
        # Total samples needed: simulation ticks + preview buffer
        total = n_ticks + N_prev

        # X reference: constant (stand still in sagittal plane)
        ref_x = np.full(total, com[0])

        # Y reference: sinusoidal sway with ramp-up
        ref_y = np.full(total, com[1])

        # ramp_start: time before sway begins [seconds]
        # Pause lets the ZMP controller stabilize its internal state
        # before receiving non-constant references.
        ramp_start = zmp_cfg.SWAY_RAMP_START

        # ramp_dur: time to reach full amplitude [seconds]
        # Gives a gentle increase. Instant full amplitude would shock
        # the controller (like a step input → overshoot).
        ramp_dur = zmp_cfg.SWAY_RAMP_DURATION

        for i in range(total):
            t = i * self.ctrl_dt
            if t > ramp_start:
                # Ramp factor: 0 → 1 over ramp_dur seconds
                ramp = min(1.0, (t - ramp_start) / ramp_dur)
                # Smoothstep for jerk-free ramp
                ramp = ramp * ramp * (3 - 2 * ramp)
                # Sinusoidal ZMP reference: y0 + A*sin(2π*f*t)
                ref_y[i] = com[1] + amplitude * ramp * np.sin(
                    2.0 * np.pi * frequency * (t - ramp_start))

        print(f"  Starting ZMP control loop ({n_ticks} ticks, {duration:.0f}s)...")

        # Get IK sync interval from config
        ik_sync_interval = self.ctrl_cfg.IK_SYNC_INTERVAL
        # Get IK iteration count for dynamic tracking
        ik_default_iter = cfg.IK.DEFAULT_ITERATIONS

        for tick in range(n_ticks):
            t_start = time.time()

            # --- ZMP preview controller step ---
            # Feed the reference trajectory from current tick onward.
            # The controller looks N_prev steps into the future.
            # Returns: desired CoM position that will produce the desired ZMP.
            com_x, _ = zmp_x.step(ref_x[tick:tick + N_prev])
            com_y, _ = zmp_y.step(ref_y[tick:tick + N_prev])
            # Full 3D CoM target: X and Y from ZMP controller, Z held at LIPM height
            com_target = np.array([com_x, com_y, z_c])

            # --- IK solve ---
            # Sync with simulation at configured interval
            # Every tick (10ms) would be ideal but is computationally wasteful
            # since the robot barely moves in one physics sub-step.
            # Configured interval is a good tradeoff.
            if tick % ik_sync_interval == 0:
                self.ik.sync_from_sim(self.data.qpos)

            # More iterations than balance phase because the
            # CoM target is now actively moving. Need better convergence
            # to track the dynamic reference accurately.
            q_target = self.ik.solve(com_target, dt=self.ctrl_dt, n_iter=ik_default_iter)

            # Step simulation with the IK-solved target
            self.step_sim(q_target, sync_viewer=True, realtime=False,
                          com_target=com_target,
                          ref_zmp_xy=np.array([ref_x[tick], ref_y[tick]]))

            # --- Monitoring ---
            # Every 200 ticks (2.0s): detailed status print
            if tick % 200 == 0:
                status = self.monitor.snapshot(self.data, "ZMP", tick, com_target)
                self.monitor.print_status(status)
                if self.monitor.is_fallen(self.data):
                    print("  *** FALLEN during ZMP control! ***")
                    return False

            if not self.viewer.is_running():
                print("  Viewer closed by user.")
                return True

            # --- Real-time pacing ---
            # Try to maintain real-time: each tick should take ctrl_dt wall-clock seconds.
            # If computation takes less, sleep the remainder.
            # If computation takes more (IK too slow), don't sleep — run as fast as possible.
            elapsed = time.time() - t_start
            sleep_time = self.ctrl_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        print("  ZMP SWAY COMPLETE.")
        return True
