"""
ZMP Preview Control for Unitree G1 Robot in MuJoCo
===================================================
Full visualization from start for debugging.

Modules:
    1. ZMPPreviewController - Kajita-style preview control
    2. G1RobotModel - Robot structure identification
    3. JointController - Actuator-adaptive control
    4. WholeBodyIK - CoM tracking with foot constraints
    5. PhaseManager - Multi-phase initialization with viz
    6. Model loading & main entry point

Requirements:
    pip install mujoco robot_descriptions numpy scipy
"""

import mujoco
import mujoco.viewer
import numpy as np
import scipy.linalg
import time
import os
import sys
from typing import Tuple, Optional
from dataclasses import dataclass

try:
    from robot_descriptions import g1_mj_description
except ImportError:
    print("Please install: pip install robot_descriptions")
    sys.exit(1)


# ================================================================
# MODULE 1: ZMP Preview Controller
# ================================================================
class ZMPPreviewController:
    """
    Kajita-style ZMP preview controller using Linear Inverted Pendulum Model (LIPM).

    The LIPM simplifies the robot as an inverted pendulum with the CoM at height z_c.
    The ZMP (Zero Moment Point) is where the net ground reaction force acts.
    The controller uses future ZMP references (preview) to generate smooth CoM motion.

    State vector x = [position, velocity, acceleration] (1D: either X or Y axis)
    Output: ZMP = position - (z_c / g) * acceleration

    Parameters:
        z_c: float
            Height of Center of Mass above ground [meters].
            Used in LIPM equation: zmp = com_pos - (z_c/g) * com_accel.
            Higher z_c means larger pendulum → slower natural dynamics.
            Typical humanoid: 0.5 - 0.9m.

        dt: float
            Control timestep [seconds].
            How often the controller updates. Must match the control loop rate.
            Typical: 0.005 - 0.02s (50-200 Hz).

        preview_time: float
            How far into the future the controller looks [seconds].
            Longer preview → smoother anticipatory motion, but more computation.
            Must be long enough to cover ~1 full step cycle.
            Typical: 1.0 - 2.0s for walking at 0.5-1.0 Hz step frequency.
            Default: 1.6s
    """

    def __init__(self, z_c: float, dt: float, preview_time: float = 1.6):
        self.dt = dt
        # N: Number of preview samples = preview_time / dt
        # e.g., 1.6s / 0.01s = 160 future reference points
        self.N = int(preview_time / dt)

        # g: Gravitational acceleration [m/s²]
        # Standard Earth gravity. Used in LIPM ZMP equation.
        g = 9.81

        # --- Continuous-time LIPM state-space ---
        # State: x = [position, velocity, jerk_integral(=acceleration)]
        # The "jerk" formulation allows smooth acceleration control.
        #
        # A: State transition matrix (continuous)
        #    dx/dt = A*x + B*u where u = jerk (rate of change of acceleration)
        #    [p_dot]     [0 1 0] [p]     [0]
        #    [p_ddot]  = [0 0 1] [v]  +  [0] * u
        #    [p_dddot]   [0 0 0] [a]     [1]
        A = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])

        # B: Input matrix (continuous). Jerk input affects only acceleration derivative.
        B = np.array([[0], [0], [1]])

        # C: Output matrix. ZMP = position - (z_c/g) * acceleration
        # This is the fundamental LIPM relationship.
        # C @ x = p - (z_c/g) * a = ZMP
        C = np.array([[1, 0, -z_c / g]])

        # --- Discretize using 2nd-order Taylor expansion ---
        # Ad ≈ I + A*dt + 0.5*A²*dt²  (more accurate than simple Euler: I + A*dt)
        # This preserves dynamics better at larger dt values.
        self.Ad = np.eye(3) + A * dt + 0.5 * (A @ A) * dt**2

        # Bd ≈ B*dt + 0.5*A*B*dt² + (1/6)*A²*B*dt³
        # Third-order term included because B only has entry in 3rd row,
        # so lower-order terms alone would miss position/velocity effects.
        self.Bd = B * dt + 0.5 * (A @ B) * dt**2 + (1.0 / 6.0) * (A @ A @ B) * dt**3

        # Cd: Discrete output matrix (same as continuous for ZMP)
        self.Cd = C

        # --- Augmented system for integral action ---
        # To eliminate steady-state ZMP tracking error, we augment the state
        # with the integral of the ZMP error.
        # Augmented state: x_tilde = [error_integral, x1, x2, x3]
        #
        # B_tilde: How input u affects augmented state
        # [C*Bd]   ← effect on ZMP error integral
        # [ Bd ]   ← effect on original state
        B_tilde = np.vstack([C @ self.Bd, self.Bd])

        # I_tilde: How the reference (desired ZMP) enters the augmented system
        # [1]      ← adds to error integral
        # [0]      ← no direct effect on state
        # [0]
        # [0]
        I_tilde = np.vstack([[1.0], np.zeros((3, 1))])

        # F_tilde: State propagation part of augmented system
        # [C*Ad]   ← how state affects ZMP (error integral update)
        # [ Ad ]   ← normal state dynamics
        F_tilde = np.vstack([C @ self.Ad, self.Ad])

        # A_tilde: Full augmented state matrix [4x4]
        # Combines error integral dynamics with state dynamics
        A_tilde = np.hstack([I_tilde, F_tilde])

        # --- LQR (Linear Quadratic Regulator) design ---
        # Q: State cost matrix [4x4]
        # Q[0,0] = 1e6: HEAVY penalty on ZMP tracking error integral.
        #   This forces the controller to aggressively eliminate steady-state error.
        #   Larger value → tighter ZMP tracking but more aggressive CoM motion.
        #   Typical range: 1e4 (loose) to 1e8 (very tight).
        # Q[1:,1:] = 0: No direct penalty on state (position/velocity/acceleration).
        #   We only care about the ZMP output, not the internal state values.
        Q = np.diag([1e6, 0.0, 0.0, 0.0])

        # R: Input cost matrix [1x1]
        # R = 1.0: Unit penalty on jerk (control effort).
        #   Larger R → smoother motion (less jerk) but slower tracking.
        #   Smaller R → faster tracking but jerkier motion.
        #   The ratio Q[0,0]/R determines the aggressiveness.
        #   Q/R = 1e6/1 = 1e6 → very responsive to ZMP error.
        R = np.array([[1.0]])

        # Solve Discrete Algebraic Riccati Equation (DARE)
        # P is the steady-state cost-to-go matrix.
        # This is the core of optimal control — finds the best tradeoff
        # between tracking error (Q) and control effort (R).
        P = scipy.linalg.solve_discrete_are(A_tilde, B_tilde, Q, R)

        # K: Optimal feedback gain [1x4]
        # u = -K @ x_tilde gives the optimal jerk input
        # K = (R + B'PB)^(-1) * B'PA  ← standard LQR formula
        K = np.linalg.inv(R + B_tilde.T @ P @ B_tilde) @ (B_tilde.T @ P @ A_tilde)

        # Split K into integral gain and state feedback gain
        # K_I: Gain on accumulated ZMP error (integral action)
        #   Ensures zero steady-state error. Higher → faster error correction.
        self.K_I = K[0, 0]

        # K_x: Gain on state [position, velocity, acceleration]
        #   Provides dynamic response / damping.
        self.K_x = K[0, 1:]

        # --- Preview gains ---
        # f[i]: Gain applied to the i-th future ZMP reference point.
        # These tell the controller how much to "anticipate" upcoming changes.
        # Earlier points (f[0]) have higher gain; later points decay toward zero.
        # The math: f[i] = X @ (Ac_tilde')^i @ P @ I_tilde
        # where Ac_tilde = A_tilde - B_tilde*K is the closed-loop system.
        Ac_tilde = A_tilde - B_tilde @ K
        X = -np.linalg.inv(R + B_tilde.T @ P @ B_tilde) @ B_tilde.T
        self.f = np.zeros(self.N)
        for i in range(self.N):
            self.f[i] = (X @ np.linalg.matrix_power(Ac_tilde.T, i) @ P @ I_tilde)[0, 0]

        # --- Internal state ---
        # x: LIPM state [position, velocity, acceleration] as 3x1 column
        self.x = np.zeros((3, 1))
        # err_sum: Accumulated ZMP tracking error (for integral action)
        self.err_sum = 0.0

        print(f"  [ZMP] z_c={z_c:.3f}m, N_preview={self.N}, K_I={self.K_I:.4f}")

    def reset(self, pos: float):
        """
        Reset controller state.

        Parameters:
            pos: Initial position [meters]. Sets position to this value
                 with zero velocity and zero acceleration.
        """
        self.x = np.array([[pos], [0.0], [0.0]])
        self.err_sum = 0.0

    def step(self, zmp_ref_seq: np.ndarray) -> Tuple[float, float]:
        """
        Execute one control step.

        Parameters:
            zmp_ref_seq: np.ndarray, shape (N,) or longer
                Future ZMP reference positions starting from current time.
                zmp_ref_seq[0] = desired ZMP at current tick
                zmp_ref_seq[1] = desired ZMP at next tick
                ...
                Must have at least self.N elements.

        Returns:
            (com_pos, com_vel): Tuple of floats
                com_pos: Desired CoM position [meters]
                com_vel: Desired CoM velocity [meters/second]
        """
        # Compute actual ZMP from current state using LIPM equation
        # zmp_actual = pos - (z_c/g) * accel = C @ x
        zmp_actual = (self.Cd @ self.x)[0, 0]

        # ZMP tracking error (actual - desired)
        err = zmp_actual - zmp_ref_seq[0]

        # Accumulate error for integral action (eliminates steady-state offset)
        self.err_sum += err

        # Preview: weighted sum of future ZMP references
        # Higher f[i] weights mean "look further ahead" for smoother anticipation
        N = min(self.N, len(zmp_ref_seq))
        preview = np.sum(self.f[:N] * zmp_ref_seq[:N])

        # Optimal control law: u = jerk
        # Three components:
        #   -K_I * err_sum:  Integral feedback (eliminate steady-state error)
        #   -K_x @ x:       State feedback (stabilize dynamics)
        #   -preview:        Feedforward from future references (anticipation)
        u = -self.K_I * self.err_sum - (self.K_x @ self.x)[0, 0] - preview

        # Propagate state: x(k+1) = Ad*x(k) + Bd*u(k)
        self.x = self.Ad @ self.x + self.Bd * u

        # Return position and velocity from state
        return float(self.x[0, 0]), float(self.x[1, 0])


# ================================================================
# MODULE 2: Robot Model Interface
# ================================================================
class G1RobotModel:
    """
    Identifies and stores G1 robot structure.

    Automatically detects:
    - Foot body IDs (for ground contact / IK constraints)
    - Actuator type (position servo vs torque)
    - Joint-to-actuator mappings
    - Appropriate PD gains per joint group
    """

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        # nq: Number of generalized coordinates (joint positions)
        #     For G1: 7 (floating base: xyz + quaternion) + N_joints
        self.nq = model.nq
        # nv: Number of degrees of freedom (joint velocities)
        #     For G1: 6 (floating base: linear + angular vel) + N_joints
        #     Note: nv < nq because quaternion (4 values) → angular vel (3 values)
        self.nv = model.nv
        # nu: Number of actuators (motors)
        self.nu = model.nu

        # --- Find foot bodies ---
        # These are the bodies whose poses we constrain in IK to keep feet planted.
        # We search by name patterns common in humanoid models.
        # The ankle_roll_link is typically the lowest link before the foot sole.
        self.left_foot_id = self._find_body([
            'left_ankle_roll_link', 'left_ankle_link', 'left_foot'
        ])
        self.right_foot_id = self._find_body([
            'right_ankle_roll_link', 'right_ankle_link', 'right_foot'
        ])

        if self.left_foot_id < 0 or self.right_foot_id < 0:
            print("  WARNING: Foot bodies not found! Listing all bodies:")
            self._list_bodies()

        # --- Detect actuator type ---
        # MuJoCo supports multiple actuator types:
        # - Position servo: ctrl = desired position, internal PD computes torque
        # - Torque/force: ctrl = raw torque applied to joint
        # G1 model uses position servos (gainprm/biasprm define internal PD).
        self.is_position_controlled = self._detect_position_actuators()

        # --- Build actuator-to-joint mappings ---
        # Each actuator drives one joint. We need the mapping to:
        # - Read current joint angle (qpos[act_to_qpos[i]])
        # - Read current joint velocity (qvel[act_to_dof[i]])
        # - Apply control to correct actuator
        self.act_to_qpos = []  # actuator index → qpos index
        self.act_to_dof = []   # actuator index → dof/qvel index
        self.act_to_jnt = []   # actuator index → joint id
        for i in range(model.nu):
            # Check that actuator transmits to a joint (not tendon/site/etc)
            if model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT:
                jnt_id = model.actuator_trnid[i, 0]
                self.act_to_qpos.append(model.jnt_qposadr[jnt_id])
                self.act_to_dof.append(model.jnt_dofadr[jnt_id])
                self.act_to_jnt.append(jnt_id)
            else:
                self.act_to_qpos.append(-1)
                self.act_to_dof.append(-1)
                self.act_to_jnt.append(-1)

        # --- PD gains for torque control mode ---
        # kp: Proportional gain [Nm/rad]. Higher → stiffer tracking.
        # kd: Derivative gain [Nm*s/rad]. Higher → more damping (less oscillation).
        # Only used if actuators are torque-type. Position servos have built-in PD.
        self.kp = np.full(model.nu, 200.0)  # Default: moderate stiffness
        self.kd = np.full(model.nu, 20.0)   # Default: moderate damping
        if not self.is_position_controlled:
            self._set_joint_gains()

        self._print_info()

    def _find_body(self, candidates: list) -> int:
        """
        Search for a body by name from a list of candidates.
        Case-insensitive partial matching.

        Parameters:
            candidates: List of name substrings to search for,
                       ordered by priority (first match wins).
        """
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if not name:
                continue
            for c in candidates:
                if c.lower() in name.lower():
                    return i
        return -1

    def _detect_position_actuators(self) -> bool:
        """
        Detect if actuators are position-controlled PD servos.

        MuJoCo position servos are defined with:
        - biastype = mjBIAS_AFFINE (bias is affine function of state)
        - gainprm[0] = kp > 0 (position gain)
        - biasprm = [0, -kp, -kd] where:
            biasprm[0] = 0 (no constant bias)
            biasprm[1] = -kp (spring toward ctrl target)
            biasprm[2] = -kd (velocity damping)

        The generated force is: f = gainprm[0]*ctrl + biasprm[0] + biasprm[1]*q + biasprm[2]*v
                                  = kp*(ctrl - q) - kd*v  (PD servo!)

        Returns:
            True if first actuator is a position servo, False otherwise.
        """
        if self.model.nu == 0:
            return False
        biastype = self.model.actuator_biastype[0]
        if biastype == mujoco.mjtBias.mjBIAS_AFFINE:
            gainprm = self.model.actuator_gainprm[0]
            biasprm = self.model.actuator_biasprm[0]
            if gainprm[0] > 0 and biasprm[1] < 0:
                return True
        return False

    def _set_joint_gains(self):
        """
        Set per-joint PD gains for torque control mode.

        Different joint groups need different gains:
        - Hip/Knee: High gains (400/40) — large masses, need stiffness for balance
        - Ankle: Medium-high (300/30) — critical for balance but lower inertia
        - Waist/Torso: Medium (200/20) — upper body stability
        - Arms/Hands: Low (80/8) — low inertia, don't need to be stiff

        The ratio kd/kp ≈ 0.1 gives critical damping for typical joint inertias.
        Underdamped (kd too low) → oscillations.
        Overdamped (kd too high) → sluggish response.
        """
        for i in range(self.model.nu):
            jnt_id = self.act_to_jnt[i]
            if jnt_id < 0:
                continue
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id) or ""
            nl = name.lower()
            if 'hip' in nl:
                self.kp[i], self.kd[i] = 400.0, 40.0
            elif 'knee' in nl:
                self.kp[i], self.kd[i] = 400.0, 40.0
            elif 'ankle' in nl:
                self.kp[i], self.kd[i] = 300.0, 30.0
            elif 'waist' in nl or 'torso' in nl:
                self.kp[i], self.kd[i] = 200.0, 20.0
            else:
                self.kp[i], self.kd[i] = 80.0, 8.0

    def _list_bodies(self):
        """Print all body names for debugging model structure."""
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            print(f"    body[{i}]: {name}")

    def _print_info(self):
        """Print identified robot structure."""
        lf_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self.left_foot_id)
        rf_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self.right_foot_id)
        print(f"  [Robot] nq={self.nq}, nv={self.nv}, nu={self.nu}")
        print(f"  [Robot] Left foot:  id={self.left_foot_id} '{lf_name}'")
        print(f"  [Robot] Right foot: id={self.right_foot_id} '{rf_name}'")
        print(f"  [Robot] Position-controlled actuators: {self.is_position_controlled}")
        if self.is_position_controlled:
            # Read the built-in servo gains from model definition
            kp = self.model.actuator_gainprm[0, 0]
            kd = -self.model.actuator_biasprm[0, 2]
            print(f"  [Robot] Built-in servo PD: kp={kp:.1f}, kd={kd:.1f}")

    def get_foot_center(self, data: mujoco.MjData) -> np.ndarray:
        """
        Midpoint between left and right foot positions.
        This approximates the center of the support polygon.
        For static balance, CoM should project onto this point.

        Returns:
            np.ndarray, shape (3,): [x, y, z] midpoint in world frame.
        """
        lf = data.xpos[self.left_foot_id]
        rf = data.xpos[self.right_foot_id]
        return (lf + rf) / 2.0

    def get_com(self, data: mujoco.MjData) -> np.ndarray:
        """
        Whole-body center of mass position.
        Calls mj_comPos to update subtree_com before reading.

        Returns:
            np.ndarray, shape (3,): [x, y, z] CoM in world frame.
        """
        mujoco.mj_comPos(self.model, data)
        return data.subtree_com[0].copy()


# ================================================================
# MODULE 3: Joint Controller
# ================================================================
class JointController:
    """
    Adapts control signal to actuator type automatically.

    For POSITION actuators (like G1):
        ctrl[i] = desired_joint_angle
        The built-in MuJoCo PD servo generates:
        torque = kp*(ctrl - q) - kd*qdot

    For TORQUE actuators:
        ctrl[i] = kp*(q_target - q) - kd*qdot + gravity_compensation
        We compute the full PD + feedforward ourselves.
    """

    def __init__(self, robot: G1RobotModel):
        self.robot = robot
        self.model = robot.model

    def set_targets_from_qpos(self, data: mujoco.MjData, q_target: np.ndarray,
                              torque_limit: float = 150.0):
        """
        Command all actuators to track q_target.

        Parameters:
            data: MjData - simulation state (reads current q, qdot, bias)
            q_target: np.ndarray, shape (nq,) - full qpos target vector
                      (only actuated joints are used; floating base entries ignored)
            torque_limit: float [Nm] - maximum torque per joint (torque mode only)
                         Prevents instability from PD overshoot.
                         150 Nm is conservative for G1-sized joints.
        """
        if self.robot.is_position_controlled:
            self._position_ctrl(data, q_target)
        else:
            self._torque_ctrl(data, q_target, torque_limit)

    def hold_default(self, data: mujoco.MjData):
        """Command robot to hold model's default pose (qpos0)."""
        self.set_targets_from_qpos(data, self.model.qpos0)

    def _position_ctrl(self, data: mujoco.MjData, q_target: np.ndarray):
        """
        Position servo mode: ctrl = desired angle.

        MuJoCo's built-in PD computes the actual torque internally as:
            torque = kp * (ctrl - q_current) - kd * qdot_current
        We just need to set ctrl to the desired position.

        CRITICAL: Do NOT send torque values here — they'd be interpreted as
        "go to position <torque_value> radians" causing catastrophic motion!
        """
        for i in range(self.model.nu):
            qpos_idx = self.robot.act_to_qpos[i]
            if qpos_idx >= 0:
                # ctrl = desired joint angle [radians]
                data.ctrl[i] = q_target[qpos_idx]

    def _torque_ctrl(self, data: mujoco.MjData, q_target: np.ndarray,
                     torque_limit: float):
        """
        Torque control mode: ctrl = PD + gravity compensation.

        Formula: tau = kp*(q_target - q) - kd*qdot + qfrc_bias

        qfrc_bias: MuJoCo's computed gravity + Coriolis forces.
                   Adding this as feedforward means the PD only needs to
                   correct for tracking error, not fight gravity.
        """
        for i in range(self.model.nu):
            qpos_idx = self.robot.act_to_qpos[i]
            dof_idx = self.robot.act_to_dof[i]
            if qpos_idx < 0 or dof_idx < 0:
                continue

            # Position error: how far from target [rad]
            err = q_target[qpos_idx] - data.qpos[qpos_idx]

            # Current joint velocity [rad/s]
            vel = data.qvel[dof_idx]

            # Gravity + Coriolis compensation [Nm]
            # qfrc_bias = M(q)*g + C(q,qdot)*qdot (gravity + velocity-dependent forces)
            bias = data.qfrc_bias[dof_idx]

            # PD control + feedforward:
            # kp*err: spring toward target (proportional)
            # -kd*vel: damping (derivative) — note: target velocity is 0
            # +bias: cancel gravity/Coriolis so PD only handles tracking
            tau = self.robot.kp[i] * err - self.robot.kd[i] * vel + bias

            # Clamp torque to prevent instability
            # Without this, large position errors → huge torques → simulation explosion
            data.ctrl[i] = np.clip(tau, -torque_limit, torque_limit)


# ================================================================
# MODULE 4: Whole-Body IK Solver
# ================================================================
class WholeBodyIK:
    """
    Whole-Body Inverse Kinematics solver.

    Solves for joint angles that:
    1. Place the CoM at a desired 3D position (primary task)
    2. Keep both feet at their captured positions/orientations (constraints)
    3. Keep non-essential joints near their reference (regularization)

    Uses damped weighted least-squares (pseudo-inverse with task priorities).

    Architecture:
    - Maintains internal "planning" state separate from simulation
    - Can sync with simulation state for tracking mode
    - Iterative solver: multiple Gauss-Newton steps per call
    """

    def __init__(self, robot: G1RobotModel):
        self.robot = robot
        self.model = robot.model
        # nv: Degrees of freedom (velocity-space dimension)
        # This is the dimension of joint velocity vector / Jacobian columns
        self.nv = robot.nv

        # Internal planning data — SEPARATE from simulation
        # This prevents IK from being corrupted if the real robot falls.
        # We solve kinematics here without physics simulation.
        self.ik_data = mujoco.MjData(self.model)

        # q_plan: Current planned joint configuration [nq]
        # This is updated after each solve() call.
        self.q_plan = self.model.qpos0.copy()

        # --- Foot constraint targets ---
        # These are captured once and held fixed during IK solving.
        # The IK will try to keep feet at these exact poses.
        self.lf_pos_target = np.zeros(3)      # Left foot position [x,y,z] meters
        self.lf_mat_target = np.eye(3)        # Left foot rotation matrix [3x3]
        self.rf_pos_target = np.zeros(3)      # Right foot position [x,y,z] meters
        self.rf_mat_target = np.eye(3)        # Right foot rotation matrix [3x3]

        # q_ref: Reference pose for regularization
        # Joints are gently pulled toward this to prevent wild arm/waist motion
        self.q_ref = self.model.qpos0.copy()

        # Flag: solver won't run until sync_from_sim is called
        self._ready = False

    def sync_from_sim(self, qpos: np.ndarray):
        """
        Synchronize IK planner state from actual simulation.

        Call this before solve() to ensure the IK starts from the robot's
        actual configuration. This is "tracking mode" — the planner follows
        the sim rather than drifting into its own trajectory.

        Parameters:
            qpos: np.ndarray, shape (nq,) - current simulation qpos
        """
        self.q_plan = qpos.copy()
        self.ik_data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.ik_data)
        self._ready = True

    def capture_foot_targets(self, qpos: Optional[np.ndarray] = None):
        """
        Lock current foot positions/orientations as IK constraints.

        After calling this, all subsequent solve() calls will try to keep
        feet at these exact poses. Call this once when feet are well-planted
        on the ground.

        Parameters:
            qpos: Optional override. If None, uses internal ik_data state.
        """
        if qpos is not None:
            self.ik_data.qpos[:] = qpos
            mujoco.mj_forward(self.model, self.ik_data)

        mujoco.mj_kinematics(self.model, self.ik_data)
        lf_id = self.robot.left_foot_id
        rf_id = self.robot.right_foot_id

        # Store current foot positions as targets
        # xpos[body_id]: body origin position in world frame [3]
        self.lf_pos_target = self.ik_data.xpos[lf_id].copy()
        self.rf_pos_target = self.ik_data.xpos[rf_id].copy()

        # Store current foot orientations as targets
        # xmat[body_id]: body rotation matrix (flattened 9 values → reshape to 3x3)
        self.lf_mat_target = self.ik_data.xmat[lf_id].reshape(3, 3).copy()
        self.rf_mat_target = self.ik_data.xmat[rf_id].reshape(3, 3).copy()

        # Reference pose for regularization
        self.q_ref = self.q_plan.copy()

    def solve(self, com_target: np.ndarray, dt: float = 0.01,
              n_iter: int = 5) -> np.ndarray:
        """
        Solve IK: find joint angles placing CoM at com_target with feet fixed.

        Uses iterative Gauss-Newton method:
        Each iteration:
            1. Compute task errors (CoM position error, foot position/orientation errors)
            2. Compute Jacobians (how joint velocities affect task-space velocities)
            3. Solve weighted least-squares for optimal joint velocity
            4. Integrate joint velocity to update joint angles
            5. Clamp to joint limits

        Parameters:
            com_target: np.ndarray, shape (3,) - desired CoM [x, y, z] in world frame
            dt: float - total time budget for this IK solve [seconds]
                Used to scale integration step: sub_dt = dt / n_iter
                Larger dt → larger steps per iteration → faster convergence but less stable
            n_iter: int - number of Gauss-Newton iterations
                More iterations → better convergence but more computation.
                3-5 is typical for real-time control.

        Returns:
            np.ndarray, shape (nq,) - solved joint configuration
        """
        if not self._ready:
            return self.q_plan.copy()

        # Start from current planned state
        self.ik_data.qpos[:] = self.q_plan
        mujoco.mj_forward(self.model, self.ik_data)

        lf_id = self.robot.left_foot_id
        rf_id = self.robot.right_foot_id

        # sub_dt: Integration step per iteration [seconds]
        # Smaller → more conservative steps, better stability
        # Total displacement ≈ velocity * dt (split across n_iter sub-steps)
        sub_dt = dt / max(n_iter, 1)

        for _ in range(n_iter):
            # Update forward kinematics to get current body poses and CoM
            mujoco.mj_kinematics(self.model, self.ik_data)
            mujoco.mj_comPos(self.model, self.ik_data)

            # --- Compute Jacobians ---
            # J_com [3 x nv]: maps joint velocities → CoM velocity
            # J_com @ qdot = com_velocity
            J_com = np.zeros((3, self.nv))
            mujoco.mj_jacSubtreeCom(self.model, self.ik_data, J_com,
                                     0)  # 0 = root body (whole robot)

            # J_lf_p [3 x nv]: maps joint velocities → left foot linear velocity
            # J_lf_r [3 x nv]: maps joint velocities → left foot angular velocity
            J_lf_p = np.zeros((3, self.nv))
            J_lf_r = np.zeros((3, self.nv))
            mujoco.mj_jacBody(self.model, self.ik_data, J_lf_p, J_lf_r, lf_id)

            # Same for right foot
            J_rf_p = np.zeros((3, self.nv))
            J_rf_r = np.zeros((3, self.nv))
            mujoco.mj_jacBody(self.model, self.ik_data, J_rf_p, J_rf_r, rf_id)

            # --- Compute task-space errors ---
            # err_com [3]: how far CoM is from target (in meters)
            err_com = com_target - self.ik_data.subtree_com[0]

            # err_lf_p [3]: left foot position error [meters]
            err_lf_p = self.lf_pos_target - self.ik_data.xpos[lf_id]

            # err_rf_p [3]: right foot position error [meters]
            err_rf_p = self.rf_pos_target - self.ik_data.xpos[rf_id]

            # err_lf_r [3]: left foot orientation error [radians, as axis-angle]
            err_lf_r = self._rot_error(
                self.ik_data.xmat[lf_id].reshape(3, 3), self.lf_mat_target)

            # err_rf_r [3]: right foot orientation error [radians, as axis-angle]
            err_rf_r = self._rot_error(
                self.ik_data.xmat[rf_id].reshape(3, 3), self.rf_mat_target)

            # --- Desired task-space velocities ---
            # v_des = kp * error  (proportional feedback in task space)
            # Higher kp → faster convergence but may overshoot
            #
            # kp_fp = 200.0: Foot position gain [1/s]
            #   High because feet MUST stay planted (ground contact)
            #   200 means: 1cm error → 2 m/s desired correction velocity
            #
            # kp_fr = 100.0: Foot rotation gain [1/s]
            #   High but less than position — orientation drift is less critical
            #   100 means: 0.01 rad error → 1 rad/s desired correction
            #
            # kp_com = 60.0: CoM gain [1/s]
            #   Lower than feet — CoM can move more slowly because it's the
            #   tracking target, not a hard constraint.
            #   60 means: 1cm error → 0.6 m/s desired correction
            kp_fp, kp_fr, kp_com = 200.0, 100.0, 60.0

            # Stack all desired velocities into one vector [15]
            # Order: [lf_pos(3), lf_rot(3), rf_pos(3), rf_rot(3), com(3)]
            v_des = np.concatenate([
                kp_fp * err_lf_p, kp_fr * err_lf_r,
                kp_fp * err_rf_p, kp_fr * err_rf_r,
                kp_com * err_com
            ])

            # Stack all Jacobians [15 x nv]
            J_stack = np.vstack([J_lf_p, J_lf_r, J_rf_p, J_rf_r, J_com])

            # --- Task weights ---
            # W [15 x 15]: diagonal weight matrix for task priority
            # Higher weight → that task is more important in the optimization.
            #
            # Foot position: 2000 — HIGHEST priority (must maintain ground contact)
            # Foot rotation: 800 — High (prevent foot from tilting/twisting)
            # CoM: 200 — Lower (it's okay to sacrifice small CoM accuracy
            #            to keep feet perfectly planted)
            #
            # Ratio matters: foot_pos/com = 2000/200 = 10x priority for feet
            W = np.diag(
                [2000.0]*3 + [800.0]*3 +   # Left foot: pos + rot
                [2000.0]*3 + [800.0]*3 +   # Right foot: pos + rot
                [200.0]*3                    # CoM
            )

            # --- Posture regularization ---
            # Prevents joints from drifting to weird configurations.
            # Without this, the IK might find solutions with wild arm poses
            # that technically satisfy CoM + foot constraints but look wrong.
            #
            # q_err [nv]: difference between current and reference joint angles
            # Computed using mj_differentiatePos which handles quaternion differences.
            q_err = np.zeros(self.nv)
            mujoco.mj_differentiatePos(self.model, q_err, 1.0,
                                       self.ik_data.qpos, self.q_ref)
            # Don't regularize floating base (DOFs 0-5) — let it move freely
            # to achieve the CoM target. Only penalize actuated joint deviations.
            q_err[:6] = 0.0

            # W_reg [nv x nv]: regularization weight per DOF
            # 5.0: Mild pull toward reference (doesn't overwhelm main tasks)
            # If too high: CoM tracking suffers because joints can't move
            # If too low: arms/waist may drift to extreme positions
            W_reg = np.eye(self.nv) * 5.0
            # Zero out floating base regularization — base must be free
            W_reg[:6, :6] = 0.0

            # --- Solve damped weighted least-squares ---
            # Problem: minimize ||W^(1/2) * (J*dq - v_des)||² + ||W_reg^(1/2) * (dq - q_err)||² + damping*||dq||²
            #
            # Normal equations: H * dq = g
            # H = J'*W*J + damping*I + W_reg
            # g = J'*W*v_des + W_reg*q_err
            #
            # damping = 5e-3: Tikhonov regularization [unit: 1/s²]
            #   Prevents singular/ill-conditioned solutions near kinematic singularities.
            #   Too small (1e-6) → near-singular solutions, joint velocity explosions
            #   Too large (1.0) → sluggish, can't reach targets
            #   5e-3 is a good balance for humanoids with ~30 DOF.
            damping = 5e-3
            H = J_stack.T @ W @ J_stack + damping * np.eye(self.nv) + W_reg
            g = J_stack.T @ W @ v_des + W_reg @ q_err

            # Solve linear system: dq = H^(-1) * g
            # dq [nv]: optimal joint velocity vector [rad/s]
            dq = np.linalg.solve(H, g)

            # --- Velocity clamp ---
            # max_vel = 4.0 [rad/s]: Maximum allowed joint velocity
            # Prevents unrealistic motions that the real robot couldn't achieve.
            # Also improves numerical stability of integration.
            # 4 rad/s ≈ 230 deg/s — fast but physically plausible for a humanoid.
            max_vel = 4.0
            scale = np.max(np.abs(dq))
            if scale > max_vel:
                dq *= max_vel / scale

            # --- Integrate joint velocities ---
            # q_new = q_old + dq * sub_dt
            # mj_integratePos handles quaternion integration correctly for floating base.
            # (Simple addition works for hinge joints but not for quaternion.)
            mujoco.mj_integratePos(self.model, self.ik_data.qpos, dq, sub_dt)

            # Clamp to physical joint limits after integration
            self._enforce_limits()

            # Recompute kinematics for next iteration
            mujoco.mj_forward(self.model, self.ik_data)

        # Store solved configuration
        self.q_plan = self.ik_data.qpos.copy()
        return self.q_plan.copy()

    def _rot_error(self, R_cur: np.ndarray, R_des: np.ndarray) -> np.ndarray:
        """
        Compute orientation error between current and desired rotation matrices.

        Returns error as an axis-angle vector [3]:
        - Direction = rotation axis
        - Magnitude = rotation angle [radians]

        Uses: R_err = R_des * R_cur^T
              angle = arccos((trace(R_err) - 1) / 2)
              axis = [R_err[2,1]-R_err[1,2], R_err[0,2]-R_err[2,0], R_err[1,0]-R_err[0,1]] / (2*sin(angle))

        Parameters:
            R_cur: [3x3] current rotation matrix
            R_des: [3x3] desired rotation matrix

        Returns:
            np.ndarray [3]: orientation error as rotation vector [rad]
        """
        # Relative rotation: "how to get from current to desired"
        R_err = R_des @ R_cur.T

        # Rotation angle from trace formula
        # trace(R) = 1 + 2*cos(angle) → angle = arccos((trace-1)/2)
        trace_val = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
        angle = np.arccos(trace_val)

        # If angle ≈ 0, no rotation needed
        if angle < 1e-6:
            return np.zeros(3)

        # Rotation axis from skew-symmetric part of R_err
        # R - R' = 2*sin(angle)*[axis]_x (skew matrix)
        axis = np.array([
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1]
        ]) / (2.0 * np.sin(angle) + 1e-10)  # 1e-10 prevents division by zero

        return axis * angle

    def _enforce_limits(self):
        """
        Clamp all limited joints to their model-defined ranges.

        Without this, the IK solver can drift joints beyond physical limits.
        When the controller then tries to track these impossible targets,
        the resulting large errors create destabilizing torques.

        Only applies to hinge (revolute) and slide (prismatic) joints.
        Free joints (floating base) have no limits.
        """
        for j in range(self.model.njnt):
            # Skip unlimited joints
            if not self.model.jnt_limited[j]:
                continue
            jtype = self.model.jnt_type[j]
            # Only clamp 1-DOF joints (hinge or slide)
            if jtype in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                adr = self.model.jnt_qposadr[j]  # Index into qpos array
                lo, hi = self.model.jnt_range[j]   # [lower_limit, upper_limit] in rad or m
                self.ik_data.qpos[adr] = np.clip(self.ik_data.qpos[adr], lo, hi)


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

    def is_fallen(self, data: mujoco.MjData, threshold: float = 0.30) -> bool:
        """
        Check if robot has fallen.

        Parameters:
            threshold: float [meters] - CoM height below which robot is "fallen"
                      0.30m is well below any humanoid's standing height.
                      G1 standing height is ~0.7m; anything below 0.30 is on the ground.
        """
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

        # --- Timing parameters ---
        # sim_dt: Physics simulation timestep [seconds]
        #   Set in model XML. Smaller → more accurate physics but slower.
        #   0.001s (1kHz) is standard for contact-rich humanoid simulation.
        self.sim_dt = model.opt.timestep

        # ctrl_dt: Control loop period [seconds]
        #   How often we run IK + update control targets.
        #   0.01s (100Hz) is typical for whole-body control.
        #   Must be ≥ sim_dt. The gap is filled by sub-stepping.
        self.ctrl_dt = 0.01

        # steps_per_ctrl: Number of physics steps per control update
        #   = ctrl_dt / sim_dt = 0.01 / 0.001 = 10
        #   Between each IK solve, the simulation runs 10 physics steps
        #   with the SAME control target held constant.
        self.steps_per_ctrl = max(1, int(self.ctrl_dt / self.sim_dt))

        print(f"  [Sim] sim_dt={self.sim_dt:.4f}s, ctrl_dt={self.ctrl_dt:.3f}s, "
              f"sub-steps={self.steps_per_ctrl}")

    def step_sim(self, q_target: np.ndarray, n_sub: Optional[int] = None,
                 sync_viewer: bool = True, realtime: bool = True):
        """
        Execute one control tick: apply control + sub-step physics + sync viewer.

        Parameters:
            q_target: np.ndarray [nq] - desired joint configuration
            n_sub: int or None - number of physics sub-steps (default: steps_per_ctrl)
            sync_viewer: bool - whether to update the viewer after stepping
            realtime: bool - (currently unused, pacing is done externally)
        """
        n = n_sub if n_sub else self.steps_per_ctrl
        for _ in range(n):
            # Apply control signal to actuators
            self.controller.set_targets_from_qpos(self.data, q_target)
            # Advance physics by one sim_dt step
            mujoco.mj_step(self.model, self.data)

        # Update viewer display
        if sync_viewer and self.viewer.is_running():
            self.viewer.sync()

    # ============================
    # PHASE 1: Settle
    # ============================
    def phase_settle(self, duration: float = 3.0) -> bool:
        """
        Let robot settle into natural standing under default pose control.

        No IK, no CoM tracking — just hold joint defaults and let gravity
        pull the robot into a stable configuration. This establishes the
        "natural" standing height and foot positions.

        Parameters:
            duration: float [seconds] - how long to settle
                     3.0s is enough for transients to die out.
                     If robot is still bouncing after 3s, there's a model issue.

        Returns:
            bool: True if robot is still standing (CoM > 0.35m), False if fallen.
        """
        print("\n" + "="*60)
        print("  PHASE 1: SETTLING (default pose hold)")
        print("="*60)

        # Total ticks at control rate
        n_ticks = int(duration / self.ctrl_dt)
        # Target: model's default joint angles
        q_default = self.model.qpos0.copy()

        for tick in range(n_ticks):
            self.step_sim(q_default)

            # Report every 0.5 seconds
            if tick % int(0.5 / self.ctrl_dt) == 0:
                status = self.monitor.snapshot(self.data, "SETTLE", tick)
                self.monitor.print_status(status)

                # threshold=0.25: very low — only catch complete collapses
                if self.monitor.is_fallen(self.data, threshold=0.25):
                    print("  *** FALLEN during settle! ***")
                    return False

            if not self.viewer.is_running():
                return False

            # Sleep for visual pacing (0.5x real-time to watch settling)
            # 0.5 multiplier: runs 2x faster than real-time for quicker startup
            time.sleep(self.ctrl_dt * 0.5)

        com = self.robot.get_com(self.data)
        print(f"  SETTLE COMPLETE: CoM height = {com[2]:.4f}m")
        # 0.35m threshold: G1 stands ~0.7m; below 0.35 means collapsed
        return com[2] > 0.35

    # ============================
    # PHASE 2: Balance acquisition
    # ============================
    def phase_balance(self, duration: float = 6.0) -> bool:
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
                     6.0s gives a gentle, stable ramp. Shorter may cause jerk.

        Returns:
            bool: True if balanced (height > 0.35m), False if fallen.
        """
        print("\n" + "="*60)
        print("  PHASE 2: BALANCE ACQUISITION (CoM -> foot center)")
        print("="*60)

        # Initialize IK from current simulation state
        self.ik.sync_from_sim(self.data.qpos)
        # Lock current foot positions as IK constraints
        # From this point, IK will keep feet at these exact poses.
        self.ik.capture_foot_targets()

        n_ticks = int(duration / self.ctrl_dt)
        # ramp_ticks: 70% of duration for the ramp, 30% for holding at target
        # This ensures smooth approach + time to verify stability at the end.
        ramp_ticks = int(n_ticks * 0.7)

        initial_com = self.robot.get_com(self.data)
        mujoco.mj_kinematics(self.model, self.data)
        foot_center = self.robot.get_foot_center(self.data)

        print(f"  Starting CoM: [{initial_com[0]:.4f}, {initial_com[1]:.4f}, {initial_com[2]:.4f}]")
        print(f"  Foot center:  [{foot_center[0]:.4f}, {foot_center[1]:.4f}, {foot_center[2]:.4f}]")

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

            # Solve IK with 3 iterations (fast, for real-time tracking)
            # n_iter=3: fewer iterations because we sync every tick anyway.
            # The small residual error is corrected next tick.
            q_target = self.ik.solve(com_target, dt=self.ctrl_dt, n_iter=3)
            self.step_sim(q_target)

            # Report every 0.5s
            if tick % int(0.5 / self.ctrl_dt) == 0:
                status = self.monitor.snapshot(self.data, "BALANCE", tick, com_target)
                self.monitor.print_status(status)
                if self.monitor.is_fallen(self.data):
                    print("  *** FALLEN during balance! ***")
                    return False

            if not self.viewer.is_running():
                return False

            # 0.3x real-time pacing (faster than real-time for quicker setup)
            time.sleep(self.ctrl_dt * 0.3)

        com = self.robot.get_com(self.data)
        fc = self.robot.get_foot_center(self.data)
        print(f"  BALANCE COMPLETE: h={com[2]:.4f}, xy_err={np.linalg.norm(com[:2]-fc[:2]):.4f}")
        return com[2] > 0.35

    # ============================
    # PHASE 3: Stability hold
    # ============================
    def phase_stability_hold(self, duration: float = 3.0) -> bool:
        """
        Hold the balanced position and verify the robot doesn't drift or fall.

        This is a sanity check before starting ZMP control.
        If the robot can't hold still for 3 seconds, ZMP sway will definitely fail.

        Parameters:
            duration: float [seconds] - hold duration
                     3.0s is enough to detect slow drift or oscillation buildup.

        Returns:
            bool: True if stable throughout, False if fallen.
        """
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

        for tick in range(n_ticks):
            # Track mode: sync every tick, solve, apply
            self.ik.sync_from_sim(self.data.qpos)
            q_target = self.ik.solve(hold_target, dt=self.ctrl_dt, n_iter=3)
            self.step_sim(q_target)

            if tick % int(0.5 / self.ctrl_dt) == 0:
                status = self.monitor.snapshot(self.data, "HOLD", tick, hold_target)
                self.monitor.print_status(status)
                max_drift = max(max_drift, status.xy_error)

                if self.monitor.is_fallen(self.data):
                    print("  *** FALLEN during hold! ***")
                    return False

            if not self.viewer.is_running():
                return False

            time.sleep(self.ctrl_dt * 0.3)

        print(f"  HOLD COMPLETE: max XY drift = {max_drift:.4f}m")
        return True

    # ============================
    # PHASE 4: ZMP Preview Sway
    # ============================
    def phase_zmp_sway(self, duration: float = 40.0,
                       amplitude: float = 0.02,
                       frequency: float = 0.2) -> bool:
        """
        Run ZMP preview control with lateral (Y-axis) sway.

        The preview controller generates a smooth CoM trajectory that
        keeps the ZMP tracking a sinusoidal reference. The robot sways
        side-to-side while maintaining balance.

        Parameters:
            duration: float [seconds] - total sway time
                     40s gives ~8 full sway cycles at 0.2Hz.

            amplitude: float [meters] - peak lateral sway distance
                      0.02m (2cm) is conservative — well within support polygon.
                      G1 foot width ~0.08m each, stance ~0.2m.
                      Support polygon half-width ~0.14m.
                      0.02m << 0.14m → very safe.
                      Can increase to 0.04-0.06m for more dramatic motion.

            frequency: float [Hz] - sway oscillation frequency
                      0.2 Hz = one full cycle every 5 seconds.
                      Natural pendulum frequency for h=0.7m: sqrt(g/h) / (2π) ≈ 0.6Hz
                      0.2Hz is well below resonance → smooth, easy to track.
                      Higher freq (0.5Hz) needs more aggressive control.

        Returns:
            bool: True if completed without falling, False otherwise.
        """
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
        zmp_x = ZMPPreviewController(z_c=z_c, dt=self.ctrl_dt)
        zmp_y = ZMPPreviewController(z_c=z_c, dt=self.ctrl_dt)
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
        # 2.0s pause lets the ZMP controller stabilize its internal state
        # before receiving non-constant references.
        ramp_start = 2.0

        # ramp_dur: time to reach full amplitude [seconds]
        # 3.0s gives a gentle increase. Instant full amplitude would shock
        # the controller (like a step input → overshoot).
        ramp_dur = 3.0

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
            # Sync with simulation every 2 ticks (20ms)
            # Every tick (10ms) would be ideal but is computationally wasteful
            # since the robot barely moves in one physics sub-step.
            # Every 2 ticks is a good tradeoff.
            if tick % 2 == 0:
                self.ik.sync_from_sim(self.data.qpos)

            # n_iter=5: More iterations than balance phase because the
            # CoM target is now actively moving. Need better convergence
            # to track the dynamic reference accurately.
            q_target = self.ik.solve(com_target, dt=self.ctrl_dt, n_iter=5)

            # Step simulation with the IK-solved target
            self.step_sim(q_target, sync_viewer=True, realtime=False)

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
    print("\n" + "="*60)
    print("  LOADING G1 MODEL")
    print("="*60)

    xml_path = g1_mj_description.MJCF_PATH
    model_dir = os.path.dirname(xml_path)
    xml_file = os.path.basename(xml_path)

    # Scene XML wraps the robot model and adds environment
    scene_xml = f"""
    <mujoco model="g1_zmp_scene">
        <!-- angle="radian": all angles in XML are in radians (not degrees) -->
        <compiler angle="radian"/>
        
        <!-- Physics solver options -->
        <option 
            timestep="0.001"
            iterations="50"
            solver="Newton"
            tolerance="1e-10"
            gravity="0 0 -9.81"
            cone="elliptic"
            impratio="10"
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
                  friction="1.0 0.005 0.001"
                  solref="0.004 1"
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

    if h < 0.3:
        # If model's default has robot at ground level, raise the floating base.
        # qpos[2] = base Z position for floating-base robots.
        # 0.75m puts G1's feet approximately at ground level.
        print(f"  Raising base to 0.75m (model default too low)...")
        data.qpos[2] = 0.75
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
    print("\n" + "#"*60)
    print("#  ZMP PREVIEW CONTROL - UNITREE G1")
    print("#  Full visualization from start")
    print("#"*60)

    # --- Load model ---
    model, data = load_g1_model()

    # *** LAUNCH VIEWER IMMEDIATELY ***
    # This is the KEY difference from the original code:
    # You can watch everything from the very first physics step.
    print("\n  Launching viewer (watch all phases)...")
    print("  Close viewer window to abort at any time.\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # --- Configure camera for good initial view ---
        # distance: How far camera is from lookat point [meters]
        #   2.5m shows full robot with some surrounding context
        viewer.cam.distance = 2.5

        # azimuth: Horizontal rotation around lookat [degrees]
        #   135° gives a 3/4 view (not pure front/side)
        viewer.cam.azimuth = 135

        # elevation: Vertical angle [degrees, negative = looking down]
        #   -20° looks slightly downward at the robot
        viewer.cam.elevation = -20

        # lookat: Point the camera is centered on [x, y, z] meters
        #   [0, 0, 0.7] = roughly chest height of standing G1
        viewer.cam.lookat[:] = [0, 0, 0.7]

        # Sync viewer to show initial configuration before any stepping
        viewer.sync()

        # Brief pause so user can see the initial state
        time.sleep(1.0)

        # Create phase manager (owns all control logic)
        manager = VisualizedPhaseManager(model, data, viewer)

        # === RUN PHASES SEQUENTIALLY ===

        # Phase 1: Settle (3 seconds)
        if not manager.phase_settle(duration=3.0):
            print("\n*** PHASE 1 FAILED: Robot cannot stand with default control ***")
            print("    Possible fixes:")
            print("    - Check if model has proper actuator definitions")
            print("    - Increase initial qpos[2] (base height)")
            print("    - Check joint limits and default pose (qpos0)")
            print("    - Verify floor contact parameters match robot feet")
            _wait_for_viewer(viewer)
            return

        # Phase 2: Balance over feet (6 seconds)
        if not manager.phase_balance(duration=6.0):
            print("\n*** PHASE 2 FAILED: Could not achieve CoM over feet ***")
            print("    Possible fixes:")
            print("    - Reduce IK aggressiveness (lower kp_com in WholeBodyIK.solve)")
            print("    - Check foot body identification (see body list above)")
            print("    - Verify IK is syncing with sim (sync_from_sim called every tick)")
            print("    - Try longer duration or smaller alpha ramp")
            _wait_for_viewer(viewer)
            return

        # Phase 3: Hold and verify (3 seconds)
        if not manager.phase_stability_hold(duration=3.0):
            print("\n*** PHASE 3 FAILED: Not stable enough for ZMP control ***")
            print("    Possible fixes:")
            print("    - Increase phase 2 duration for better convergence")
            print("    - Check if foot constraints are holding (feet sliding?)")
            print("    - Verify PD gains are appropriate for this model")
            _wait_for_viewer(viewer)
            return

        # Phase 4: ZMP sway!
        print("\n  All stability phases passed! Starting ZMP sway...")
        manager.phase_zmp_sway(
            duration=40.0,       # 40 seconds of sway
            amplitude=0.02,      # 2cm lateral sway (conservative start)
            frequency=0.2        # 0.2 Hz = 5 second period
        )

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
        # 50ms sleep = 20fps update rate for idle viewer
        # Low enough to be responsive to user closing window
        time.sleep(0.05)


# ================================================================
if __name__ == "__main__":
    main()