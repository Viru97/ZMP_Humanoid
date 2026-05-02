import mujoco
import numpy as np
from robot_model import G1RobotModel


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
