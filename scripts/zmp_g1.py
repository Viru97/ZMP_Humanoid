import mujoco
import mujoco.viewer
import numpy as np
import scipy.linalg
import time
import os

try:
    from robot_descriptions import g1_mj_description
except ImportError:
    print("Please install the required library first: pip install robot_descriptions")
    exit()


# ==========================================
# 1. ZMP PREVIEW CONTROLLER (LQR)
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
# 2. MAIN SIMULATION PIPELINE
# ==========================================
def main():
    # Setup MuJoCo with Unitree G1 and add a floor!
    # Convert backslashes for Windows path compatibility within the XML string
    xml_path = g1_mj_description.MJCF_PATH.replace('\\', '/')
    model_dir = os.path.dirname(g1_mj_description.MJCF_PATH)

    scene_xml = f"""
    <mujoco>
        <!-- Include the raw robot model -->
        <include file="{xml_path}"/>

        <!-- Add a physical environment -->
        <worldbody>
            <light pos="0 0 3" dir="0 0 -1" directional="true"/>
            <geom name="floor" type="plane" size="5 5 0.1" rgba="0.8 0.9 0.8 1" contype="1" conaffinity="1"/>
        </worldbody>
    </mujoco>
    """

    # Temporarily change the working directory so MuJoCo can find the robot's relative STL meshes
    current_cwd = os.getcwd()
    os.chdir(model_dir)
    try:
        model = mujoco.MjModel.from_xml_string(scene_xml)
    finally:
        os.chdir(current_cwd)  # Always switch back, even if it fails

    data = mujoco.MjData(model)

    # Lift the robot 80cm into the air so it spawns above the newly created floor
    if model.nq >= 7:  # Ensure the robot has a free joint (floating base)
        data.qpos[2] = 0.8

    # Control Parameters
    dt = 0.01  # Control rate (100 Hz)
    z_c = 0.7  # Approximate target CoM height for G1
    preview_time = 1.5
    zmp_ctrl = ZMPPreviewController(z_c=z_c, dt=dt, preview_time=preview_time)

    # Initialize physics
    mujoco.mj_forward(model, data)

    # Fetch number of actuators
    nu = model.nu
    print(f"Loaded Unitree G1 with {nu} actuators.")

    # Set an initial control vector (holding 0 posture)
    initial_ctrl = np.zeros(nu)
    data.ctrl[:] = initial_ctrl

    N_ticks = 10000

    # --- Run Simulation Viewer ---
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for i in range(N_ticks):
            step_start = time.time()

            # --- 3D ZMP + IK INTEGRATION PLACEHOLDER ---
            # 1. Get desired 3D CoM from zmp_ctrl (we now need X and Y lateral ZMP)
            # zmp_window_x = ...
            # com_x, _ = zmp_ctrl_x.step(zmp_window_x)
            # com_y, _ = zmp_ctrl_y.step(zmp_window_y)

            # 2. Use 3D Whole-Body Inverse Kinematics (Pinocchio / Jacobian)
            # to map (com_x, com_y, com_z) + Foot Placements into the 29 joint angles.
            # q_target = solve_whole_body_ik(com_target, foot_target)

            # 3. Apply to MuJoCo Actuators
            # data.ctrl[:] = q_target

            # For now, we hold the initial posture to prevent immediate collapse
            data.ctrl[:] = initial_ctrl

            # 4. Step Physics Engine (Control rate is 100Hz, Physics is typically 1000Hz)
            for _ in range(10):
                mujoco.mj_step(model, data)

            viewer.sync()

            # Maintain real-time execution
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()