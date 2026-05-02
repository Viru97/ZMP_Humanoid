import mujoco
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from config import cfg


@dataclass
class g1BodyIDs:
    """Struct to hold important MuJoCo body IDs for quick access."""
    pelvis: int
    left_foot: int
    right_foot: int


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

        # Get robot configuration from config
        robot_cfg = cfg.ROBOT

        # --- Find foot bodies ---
        # These are the bodies whose poses we constrain in IK to keep feet planted.
        # We search by name patterns common in humanoid models.
        # The ankle_roll_link is typically the lowest link before the foot sole.
        self.left_foot_id = self._find_body(list(robot_cfg.LEFT_FOOT_CANDIDATES))
        self.right_foot_id = self._find_body(list(robot_cfg.RIGHT_FOOT_CANDIDATES))

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
        self.kp = np.full(model.nu, robot_cfg.KP_JOINT_DEFAULT)  # Default: moderate stiffness
        self.kd = np.full(model.nu, robot_cfg.KD_JOINT_DEFAULT)  # Default: moderate damping
        if not self.is_position_controlled:
            self._set_joint_gains()

        self._print_info()

    def _find_body(self, candidates: List[str]) -> int:
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
        - Hip/Knee: High gains — large masses, need stiffness for balance
        - Ankle: Medium-high — critical for balance but lower inertia
        - Waist/Torso: Medium — upper body stability
        - Arms/Hands: Low — low inertia, don't need to be stiff

        The ratio kd/kp ≈ 0.1 gives critical damping for typical joint inertias.
        Underdamped (kd too low) → oscillations.
        Overdamped (kd too high) → sluggish response.
        """
        # Get gains from config
        robot_cfg = cfg.ROBOT

        for i in range(self.model.nu):
            jnt_id = self.act_to_jnt[i]
            if jnt_id < 0:
                continue
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id) or ""
            nl = name.lower()
            if 'hip' in nl:
                self.kp[i], self.kd[i] = robot_cfg.KP_HIP, robot_cfg.KD_HIP
            elif 'knee' in nl:
                self.kp[i], self.kd[i] = robot_cfg.KP_KNEE, robot_cfg.KD_KNEE
            elif 'ankle' in nl:
                self.kp[i], self.kd[i] = robot_cfg.KP_ANKLE, robot_cfg.KD_ANKLE
            elif 'waist' in nl or 'torso' in nl:
                self.kp[i], self.kd[i] = robot_cfg.KP_WAIST, robot_cfg.KD_WAIST
            else:
                self.kp[i], self.kd[i] = robot_cfg.KP_ARM, robot_cfg.KD_ARM

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