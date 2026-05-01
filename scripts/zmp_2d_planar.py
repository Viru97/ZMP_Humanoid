import mujoco
import mujoco.viewer
import numpy as np
import scipy.linalg
import time

# ==========================================
# 1. EMBEDDED ROBOT MODEL (MJCF / XML)
# ==========================================
# A planar biped constrained to the X-Z plane.
# It has a Torso, and two 3-DOF legs (Hip, Knee, Ankle).
BIPED_XML = """
<mujoco model="planar_biped">
    <option timestep="0.001" gravity="0 0 -9.81"/>

    <default>
        <!-- Collision Filtering: Robot is group 2, Floor is group 1. Prevents leg self-collisions. -->
        <geom friction="1 0.1 0.1" contype="2" conaffinity="1"/>

        <!-- Added armature inertia (0.1) to heavily stabilize the QACC solver -->
        <joint damping="1.0" armature="0.1"/>
        <motor ctrlrange="-100 100" ctrllimited="true"/>

        <!-- Optimized PD gains for critical damping given the new armature inertia -->
        <position kp="1500" kv="50"/> 
    </default>

    <asset>
        <material name="mat_torso" rgba="0.2 0.6 0.8 1"/>
        <material name="mat_leg_r" rgba="0.8 0.2 0.2 1"/>
        <material name="mat_leg_l" rgba="0.2 0.8 0.2 1"/>
    </asset>

    <worldbody>
        <light pos="0 0 3" dir="0 0 -1" directional="true"/>
        <!-- Floor matches with robot's conaffinity to allow foot-ground contact -->
        <geom name="floor" type="plane" size="5 1 0.1" rgba="0.8 0.9 0.8 1" contype="1" conaffinity="2"/>

        <!-- Torso -->
        <body name="torso" pos="0 0 0.6">
            <!-- Constrain to 2D (Sagittal Plane X-Z) -->
            <joint name="root_x" type="slide" axis="1 0 0" limited="false" damping="0"/>
            <joint name="root_z" type="slide" axis="0 0 1" limited="false" damping="0"/>
            <!-- No root_y rotation (pitch), forces torso to stay perfectly upright for simple LIPM -->

            <geom name="torso_geom" type="box" size="0.05 0.1 0.1" mass="10.0" material="mat_torso"/>

            <!-- RIGHT LEG -->
            <body name="r_thigh" pos="0 -0.05 -0.1">
                <joint name="r_hip" type="hinge" axis="0 1 0"/>
                <geom name="r_thigh_geom" type="capsule" fromto="0 0 0  0 0 -0.3" size="0.03" mass="1.0" material="mat_leg_r"/>

                <body name="r_shin" pos="0 0 -0.3">
                    <joint name="r_knee" type="hinge" axis="0 1 0"/>
                    <geom name="r_shin_geom" type="capsule" fromto="0 0 0  0 0 -0.3" size="0.02" mass="0.5" material="mat_leg_r"/>

                    <body name="r_foot" pos="0 0 -0.3">
                        <joint name="r_ankle" type="hinge" axis="0 1 0"/>
                        <geom name="r_foot_geom" type="box" pos="0.02 0 -0.02" size="0.08 0.04 0.02" mass="0.2" material="mat_leg_r"/>
                    </body>
                </body>
            </body>

            <!-- LEFT LEG -->
            <body name="l_thigh" pos="0 0.05 -0.1">
                <joint name="l_hip" type="hinge" axis="0 1 0"/>
                <geom name="l_thigh_geom" type="capsule" fromto="0 0 0  0 0 -0.3" size="0.03" mass="1.0" material="mat_leg_l"/>

                <body name="l_shin" pos="0 0 -0.3">
                    <joint name="l_knee" type="hinge" axis="0 1 0"/>
                    <geom name="l_shin_geom" type="capsule" fromto="0 0 0  0 0 -0.3" size="0.02" mass="0.5" material="mat_leg_l"/>

                    <body name="l_foot" pos="0 0 -0.3">
                        <joint name="l_ankle" type="hinge" axis="0 1 0"/>
                        <geom name="l_foot_geom" type="box" pos="0.02 0 -0.02" size="0.08 0.04 0.02" mass="0.2" material="mat_leg_l"/>
                    </body>
                </body>
            </body>
        </body>
    </worldbody>

    <actuator>
        <position name="r_hip_act" joint="r_hip"/>
        <position name="r_knee_act" joint="r_knee"/>
        <position name="r_ankle_act" joint="r_ankle"/>
        <position name="l_hip_act" joint="l_hip"/>
        <position name="l_knee_act" joint="l_knee"/>
        <position name="l_ankle_act" joint="l_ankle"/>
    </actuator>
</mujoco>
"""


# ==========================================
# 2. ZMP PREVIEW CONTROLLER (LQR)
# ==========================================
class ZMPPreviewController:
    """
    Kajita's LQR Preview Controller for the Linear Inverted Pendulum Model (LIPM).
    Generates a stable CoM trajectory given future ZMP setpoints.
    """

    def __init__(self, z_c, dt, preview_time=1.5):
        self.dt = dt
        self.N = int(preview_time / dt)
        g = 9.81

        # 1D Continuous LIPM State Space
        A = np.array([[0, 1, 0],
                      [0, 0, 1],
                      [0, 0, 0]])
        B = np.array([[0], [0], [1]])
        C = np.array([[1, 0, -z_c / g]])

        # Discretize using exact exponential
        self.Ad = np.eye(3) + A * dt + 0.5 * np.dot(A, A) * dt ** 2
        self.Bd = B * dt + 0.5 * np.dot(A, B) * dt ** 2 + (1 / 6) * np.linalg.matrix_power(A, 2).dot(B) * dt ** 3
        self.Cd = C

        # Augmented Error System (Kajita 2003 formulation)
        B_tilde = np.vstack([self.Cd.dot(self.Bd), self.Bd])
        I_tilde = np.vstack([[1], np.zeros((3, 1))])
        F_tilde = np.vstack([self.Cd.dot(self.Ad), self.Ad])
        A_tilde = np.hstack([I_tilde, F_tilde])

        # LQR Weights
        Q = np.zeros((4, 4));
        Q[0, 0] = 1e6  # High penalty on ZMP tracking error
        R = np.array([[1.0]])  # Low penalty on jerk (input)

        # Solve Discrete Algebraic Riccati Equation (DARE)
        P = scipy.linalg.solve_discrete_are(A_tilde, B_tilde, Q, R)

        # Compute gains
        K = np.linalg.inv(R + B_tilde.T.dot(P).dot(B_tilde)).dot(B_tilde.T).dot(P).dot(A_tilde)
        self.K_I = K[0, 0]
        self.K_x = K[0, 1:]

        # Compute Preview Feedforward Gains
        self.f = np.zeros(self.N)
        Ac_tilde = A_tilde - B_tilde.dot(K)
        X = -np.linalg.inv(R + B_tilde.T.dot(P).dot(B_tilde)).dot(B_tilde.T)
        for i in range(self.N):
            self.f[i] = X.dot(np.linalg.matrix_power(Ac_tilde.T, i)).dot(P).dot(I_tilde)[0, 0]

        # State Initialization: [pos, vel, accel]
        self.x = np.zeros((3, 1))
        self.err_sum = 0

    def step(self, zmp_ref_seq):
        """ Progresses the CoM physics one control step """
        zmp_actual = self.Cd.dot(self.x)[0, 0]
        zmp_err = zmp_actual - zmp_ref_seq[0]
        self.err_sum += zmp_err

        # Apply preview window array
        preview_term = np.sum(self.f * zmp_ref_seq[:self.N])

        u = -self.K_I * self.err_sum - self.K_x.dot(self.x)[0] - preview_term
        self.x = self.Ad.dot(self.x) + self.Bd * u

        return self.x[0, 0], self.x[1, 0]  # Return X pos and velocity


# ==========================================
# 3. ANALYTICAL INVERSE KINEMATICS
# ==========================================
def solve_leg_ik(dx, dz, L1=0.3, L2=0.3):
    """
    Computes hip, knee, and ankle angles for a 3-link planar leg.
    dx, dz: target position of the foot relative to the hip.
    """
    D = np.sqrt(dx ** 2 + dz ** 2)
    D = np.clip(D, 0.01, L1 + L2 - 0.001)  # Prevent singularity

    # Cosine rule for knee
    cos_knee = (L1 ** 2 + L2 ** 2 - D ** 2) / (2 * L1 * L2)
    gamma = np.arccos(cos_knee)

    # MuJoCo signs: +Y rotation moves link forward (+X)
    knee_angle = -(np.pi - gamma)  # Bend backwards

    # Cosine rule for hip
    alpha = np.arctan2(dx, -dz)  # Angle of foot relative to downward vertical
    cos_beta = (L1 ** 2 + D ** 2 - L2 ** 2) / (2 * L1 * D)
    beta = np.arccos(cos_beta)

    hip_angle = alpha + beta  # Swing thigh forward

    # Ankle to keep foot flat (parallel to ground)
    ankle_angle = -(hip_angle + knee_angle)

    return hip_angle, knee_angle, ankle_angle


# ==========================================
# 4. MAIN SIMULATION PIPELINE
# ==========================================
def main():
    # Setup MuJoCo
    model = mujoco.MjModel.from_xml_string(BIPED_XML)
    data = mujoco.MjData(model)

    # Control Parameters
    dt = 0.01  # Control rate (100 Hz)
    z_c = 0.6  # Target CoM height
    preview_time = 1.5
    zmp_ctrl = ZMPPreviewController(z_c=z_c, dt=dt, preview_time=preview_time)

    # Initialize robot in a bent-knee stance
    base_foot_z = 0.04  # Ankle height when the 0.04m thick sole is perfectly on the floor
    init_hip_z = z_c - 0.1
    init_q = solve_leg_ik(0.0, base_foot_z - init_hip_z)

    # Spawn the robot 2cm in the air so it gently settles onto the ground
    data.qpos[1] = 0.02

    data.qpos[2:5] = init_q
    data.qpos[5:8] = init_q
    data.ctrl[0:3] = init_q
    data.ctrl[3:6] = init_q
    mujoco.mj_forward(model, data)

    # --- Trajectory Generation ---
    # We will plan 6 forward steps
    step_len = 0.15
    t_step = 0.6  # Total time per step
    N_ticks = 1000  # Total control loop ticks (10 seconds)

    zmp_ref = np.zeros(N_ticks + zmp_ctrl.N)
    foot_x = {'l': np.zeros(N_ticks), 'r': np.zeros(N_ticks)}
    foot_z = {'l': np.full(N_ticks, base_foot_z), 'r': np.full(N_ticks, base_foot_z)}

    for i in range(N_ticks):
        time_sec = i * dt

        # Initial wait phase (stand still)
        if time_sec < 1.0:
            zmp_ref[i] = 0.0
            continue

        # Walking Phase
        phase_time = time_sec - 1.0
        step_idx = int(phase_time / t_step)
        t_in_step = phase_time % t_step

        is_left_swing = (step_idx % 2 == 0)

        # Max steps to take
        if step_idx < 6:
            target_x = (step_idx + 1) * step_len
            stance_x = step_idx * step_len
            zmp_ref[i] = stance_x

            # Swing Foot Trajectory (Sine curve for Z, Linear for X)
            swing_z = base_foot_z + 0.05 * np.sin(np.pi * (t_in_step / t_step))  # Max height 5cm
            swing_x = stance_x - step_len + (step_len * (t_in_step / t_step))

            if is_left_swing:
                foot_x['l'][i], foot_z['l'][i] = swing_x, swing_z
                foot_x['r'][i], foot_z['r'][i] = stance_x, base_foot_z
            else:
                foot_x['r'][i], foot_z['r'][i] = swing_x, swing_z
                foot_x['l'][i], foot_z['l'][i] = stance_x, base_foot_z
        else:
            # End of steps, hold position
            zmp_ref[i] = 6 * step_len
            foot_x['l'][i], foot_z['l'][i] = 6 * step_len, base_foot_z
            foot_x['r'][i], foot_z['r'][i] = 6 * step_len, base_foot_z

    # Fill the preview tail to prevent out-of-bounds
    zmp_ref[N_ticks:] = zmp_ref[N_ticks - 1]

    # --- Run Simulation Viewer ---
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for i in range(N_ticks):
            step_start = time.time()

            # 1. ZMP Preview Control: Get desired CoM
            zmp_window = zmp_ref[i: i + zmp_ctrl.N]
            com_x, _ = zmp_ctrl.step(zmp_window)

            # 2. Extract Desired Foot Positions
            lx, lz = foot_x['l'][i], foot_z['l'][i]
            rx, rz = foot_x['r'][i], foot_z['r'][i]

            # 3. Inverse Kinematics
            # Target foot pos relative to the hip (which follows CoM X and target height z_c)
            # Leg starts 0.1m below the torso root (see XML), so hip Z is actually z_c - 0.1
            hip_z = z_c - 0.1

            l_rel_x, l_rel_z = lx - com_x, lz - hip_z
            r_rel_x, r_rel_z = rx - com_x, rz - hip_z

            l_q = solve_leg_ik(l_rel_x, l_rel_z)
            r_q = solve_leg_ik(r_rel_x, r_rel_z)

            # 4. Apply to MuJoCo Actuators
            data.ctrl[0:3] = r_q
            data.ctrl[3:6] = l_q

            # 5. Step Physics Engine (Control rate is 100Hz, Physics is 1000Hz)
            for _ in range(10):
                mujoco.mj_step(model, data)

            viewer.sync()

            # Maintain real-time execution
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()