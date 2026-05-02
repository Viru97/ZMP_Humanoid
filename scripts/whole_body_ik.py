import mujoco
import numpy as np
from typing import Optional
from robot_model import G1RobotModel
from config import cfg


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

        # Get IK configuration from config
        self.ik_cfg = cfg.IK

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
              n_iter: int = None) -> np.ndarray:
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
                Default from config.IK.DEFAULT_ITERATIONS is typical for real-time control.

        Returns:
            np.ndarray, shape (nq,) - solved joint configuration
        """
        if n_iter is None:
            n_iter = self.ik_cfg.DEFAULT_ITERATIONS

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
        sub_dt = 0.001  # fixed 1ms integration step (was: dt / n_iter which was too large)

        # Get IK parameters from config
        kp_fp = self.ik_cfg.KP_FOOT_POSITION
        kp_fr = self.ik_cfg.KP_FOOT_ROTATION
        kp_com = self.ik_cfg.KP_COM

        w_fp = self.ik_cfg.WEIGHT_FOOT_POSITION
        w_fr = self.ik_cfg.WEIGHT_FOOT_ROTATION
        w_com = self.ik_cfg.WEIGHT_COM

        damping = self.ik_cfg.DAMPING
        reg_weight = self.ik_cfg.JOINT_REG_WEIGHT
        max_vel = self.ik_cfg.MAX_JOINT_VELOCITY

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
            # kp_fp: Foot position gain [1/s]
            #   High because feet MUST stay planted (ground contact)
            #   200 means: 1cm error → 2 m/s desired correction velocity
            #
            # kp_fr: Foot rotation gain [1/s]
            #   High but less than position — orientation drift is less critical
            #   100 means: 0.01 rad error → 1 rad/s desired correction
            #
            # kp_com: CoM gain [1/s]
            #   Lower than feet — CoM can move more slowly because it's the
            #   tracking target, not a hard constraint.
            #   60 means: 1cm error → 0.6 m/s desired correction

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
            # Foot position: HIGHEST priority (must maintain ground contact)
            # Foot rotation: High (prevent foot from tilting/twisting)
            # CoM: Lower (it's okay to sacrifice small CoM accuracy
            #            to keep feet perfectly planted)
            #
            # Ratio matters: foot_pos/com = priority for feet over CoM
            W = np.diag(
                [w_fp]*3 + [w_fr]*3 +   # Left foot: pos + rot
                [w_fp]*3 + [w_fr]*3 +   # Right foot: pos + rot
                [w_com]*3                 # CoM
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
            # Mild pull toward reference (doesn't overwhelm main tasks)
            # If too high: CoM tracking suffers because joints can't move
            # If too low: arms/waist may drift to extreme positions
            W_reg = np.eye(self.nv) * reg_weight
            # Zero out floating base regularization — base must be free
            W_reg[:6, :6] = 0.0

            # --- Solve damped weighted least-squares ---
            # Problem: minimize ||W^(1/2) * (J*dq - v_des)||² + ||W_reg^(1/2) * (dq - q_err)||² + damping*||dq||²
            #
            # Normal equations: H * dq = g
            # H = J'*W*J + damping*I + W_reg
            # g = J'*W*v_des + W_reg*q_err
            #
            # damping: Tikhonov regularization [unit: 1/s²]
            #   Prevents singular/ill-conditioned solutions near kinematic singularities.
            #   Too small (1e-6) → near-singular solutions, joint velocity explosions
            #   Too large (1.0) → sluggish, can't reach targets
            #   Default from config is a good balance for humanoids with ~30 DOF.
            H = J_stack.T @ W @ J_stack + damping * np.eye(self.nv) + W_reg
            g = J_stack.T @ W @ v_des + W_reg @ q_err

            # Solve linear system: dq = H^(-1) * g
            # dq [nv]: optimal joint velocity vector [rad/s]
            dq = np.linalg.solve(H, g)

            # --- Velocity clamp ---
            # max_vel: Maximum allowed joint velocity
            # Prevents unrealistic motions that the real robot couldn't achieve.
            # Also improves numerical stability of integration.
            # 4 rad/s ≈ 230 deg/s — fast but physically plausible for a humanoid.
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
