import numpy as np
import scipy.linalg


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