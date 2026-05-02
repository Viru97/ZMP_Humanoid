# ================================================================
# PROJECT CONFIGURATION
# ================================================================
"""
Centralized configuration for ZMP Humanoid Robot Project.

This module defines all simulation, robot, and controller parameters
to ensure consistent settings across the entire project.

Usage:
    from config import Config, cfg, get_config
    # Access parameters: cfg.SIMULATION.DT
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict
from enum import Enum


# ================================================================
# ENUMERATIONS
# ================================================================
class ControlMode(Enum):
    """Robot actuation control modes."""
    POSITION = "position"  # MuJoCo built-in PD servos
    TORQUE = "torque"  # Custom PD + gravity compensation


class WalkingPhase(Enum):
    """Robot walking phases."""
    SETTLE = "settle"  # Initial settling on ground
    BALANCE = "balance"  # Standing balance
    PREPARE = "prepare"  # Preparing for step
    STEP = "step"  # Single support phase
    DOUBLE_SUPPORT = "double_support"  # Both feet on ground


# ================================================================
# CONFIGURATION CLASSES
# ================================================================

@dataclass
class SimulationConfig:
    """Simulation engine parameters."""
    DT: float = 0.001  # Physics simulation timestep [s] - 1000 Hz default
    # Determines accuracy of physics calculations (lower = more accurate but slower)
    CONTROL_DT: float = 0.01  # Control loop timestep [s] - 100 Hz default
    # How often the ZMP controller and phase manager update commands
    # Must be an integer multiple of DT (default: 10 physics steps per control step)
    SIMULATION_TIME: float = 20.0  # Total simulation time [s]
    GRAVITY: Tuple[float, float, float] = (0, 0, -9.81)  # Gravity vector [m/s²]
    VIEWER_FPS: int = 60  # Target FPS for visualization
    MAX_STEPS: int = 2000  # Maximum simulation steps to prevent infinite loops

    # MuJoCo solver options
    SOLVER_ITERATIONS: int = 50  # Max solver iterations per step
    SOLVER_TOLERANCE: float = 1e-10  # Solver convergence tolerance
    SOLVER_TYPE: str = "Newton"  # Solver type: "Newton" or "PGS"
    CONE_TYPE: str = "elliptic"  # Friction cone: "elliptic" or "pyramidal"
    IMPRATIO: float = 10.0  # Contact impedance ratio (higher = stiffer contacts)

    # Ground contact parameters
    GROUND_FRICTION: Tuple[float, float, float] = (1.0, 0.005, 0.001)  # mu_slide, mu_spin, mu_roll
    GROUND_SOLREF: Tuple[float, float] = (0.004, 1.0)  # timeconst, dampratio

    def __post_init__(self):
        """Compute derived values."""
        # Compute steps per control cycle
        self.STEPS_PER_CONTROL = max(1, int(self.CONTROL_DT / self.DT))


@dataclass
class RobotModelConfig:
    """Robot model and physical parameters based on G1 humanoid structure."""
    NAME: str = "G1"
    XML_PATH: str = "assets/g1.xml"  # Path to MuJoCo robot model XML file

    # Robot dimensions
    ROBOT_HEIGHT: float = 1.5  # [m] approximate robot height from base to head
    COM_HEIGHT: float = 0.85  # [m] Center of Mass height above ground
    # Critical for LIPM (Linear Inverted Pendulum Model) equations
    # Typical range for humanoids: 0.5-0.9m
    INITIAL_BASE_HEIGHT: float = 0.75  # [m] Initial base height when model default is too low

    # Mass properties
    TOTAL_MASS: float = 45.0  # [kg] total robot mass

    # Fall detection
    FALL_HEIGHT_THRESHOLD: float = 0.30  # [m] CoM height below which robot is "fallen"
    MIN_STANDING_HEIGHT: float = 0.35  # [m] Minimum CoM height for valid standing

    # Joint limits (radians) - Based on G1 robot specifications
    JOINT_LIMITS: Optional[Dict[str, Tuple[float, float]]] = None  # Will be populated from robot model or defaults below

    # Joint gains for position control (used by JointController)
    KP_DEFAULT: float = 1500.0  # [Nm/rad] Position gain for built-in MuJoCo PD servos
    KD_DEFAULT: float = 50.0  # [Nm/(rad/s)] Derivative gain for built-in PD servos

    # Joint gains for torque control (used when is_position_controlled=False)
    KP_TORQUE: float = 50.0  # [Nm/rad] Proportional gain for custom PD controller
    KD_TORQUE: float = 10.0  # [Nm/(rad/s)] Derivative gain for custom PD controller

    # Default per-joint gains for torque control mode
    KP_HIP: float = 400.0
    KD_HIP: float = 40.0
    KP_KNEE: float = 400.0
    KD_KNEE: float = 40.0
    KP_ANKLE: float = 300.0
    KD_ANKLE: float = 30.0
    KP_WAIST: float = 200.0
    KD_WAIST: float = 20.0
    KP_ARM: float = 80.0
    KD_ARM: float = 8.0
    KP_JOINT_DEFAULT: float = 200.0
    KD_JOINT_DEFAULT: float = 20.0

    # Torque and velocity limits for safety
    MAX_TORQUE: float = 150.0  # [Nm] maximum joint torque (conservative limit for G1)
    MAX_VELOCITY: float = 10.0  # [rad/s] maximum joint velocity

    # Body name candidates for foot detection
    LEFT_FOOT_CANDIDATES: Tuple[str, ...] = ('left_ankle_roll_link', 'left_ankle_link', 'left_foot')
    RIGHT_FOOT_CANDIDATES: Tuple[str, ...] = ('right_ankle_roll_link', 'right_ankle_link', 'right_foot')

    def __post_init__(self):
        # Default joint limits for G1 robot (based on analysis of robot_model.py)
        if self.JOINT_LIMITS is None:
            self.JOINT_LIMITS = {
                # Left leg joints
                "left_hip_x": (-0.52, 0.52),  # Hip yaw
                "left_hip_y": (-0.44, 0.8),  # Hip pitch
                "left_hip_z": (-0.44, 0.44),  # Hip roll
                "left_knee": (0.0, 2.25),  # Knee pitch
                "left_ankle_x": (-0.87, 0.52),  # Ankle pitch
                "left_ankle_y": (-0.52, 0.52),  # Ankle roll

                # Right leg joints
                "right_hip_x": (-0.52, 0.52),
                "right_hip_y": (-0.44, 0.8),
                "right_hip_z": (-0.44, 0.44),
                "right_knee": (0.0, 2.25),
                "right_ankle_x": (-0.87, 0.52),
                "right_ankle_y": (-0.52, 0.52),

                # Waist joints
                "waist_yaw": (-2.62, 2.62),
                "waist_pitch": (-0.52, 0.52),
                "waist_roll": (-0.26, 0.26),

                # Arm joints
                "left_shoulder_pitch": (-2.97, 2.97),
                "left_shoulder_roll": (-0.52, 1.66),
                "left_shoulder_yaw": (-2.62, 2.62),
                "left_elbow": (-0.52, 2.97),

                "right_shoulder_pitch": (-2.97, 2.97),
                "right_shoulder_roll": (-1.66, 0.52),
                "right_shoulder_yaw": (-2.62, 2.62),
                "right_elbow": (-0.52, 2.97),
            }


@dataclass
class ZMPControllerConfig:
    """ZMP (Zero Moment Point) controller parameters based on Kajita's preview control."""
    # LIPM (Linear Inverted Pendulum Model) parameters
    Z_C: float = 0.85  # [m] CoM height above ground for LIPM model
    # Determines natural pendulum dynamics: z_c/g appears in ZMP equation
    # Higher z_c = larger pendulum = slower dynamics (more stable but less responsive)

    PREVIEW_TIME: float = 1.6  # [s] Preview horizon for future ZMP references
    # Longer preview = smoother motion but more computation
    # Should cover ~1 full step cycle (typical: 1.0-2.0s at 0.5-1.0 Hz)

    # Control timing
    DT: float = 0.01  # [s] Control loop timestep (default: 100 Hz)

    # LQR (Linear Quadratic Regulator) weights
    Q_ZMP_ERROR: float = 1e6  # Weight on ZMP tracking error integral
    # Higher value = tighter ZMP tracking but more aggressive CoM motion
    # Typical range: 1e4 (loose) to 1e8 (very tight)
    Q_STATE: float = 0.0  # Weight on state (position/velocity/acceleration) - not used currently
    R_JERK: float = 1.0  # Weight on control effort (jerk = derivative of acceleration)
    # Higher value = smoother motion (less jerk) but slower tracking
    # Ratio Q[0,0]/R determines overall aggressiveness (default: 1e6)

    # Walking parameters (used by phase_manager and trajectory generation)
    STEP_LENGTH: float = 0.15  # [m] target step length for walking gait
    STEP_TIME: float = 0.6  # [s] time per step cycle (step frequency ≈ 1.67 Hz)
    SWING_HEIGHT: float = 0.05  # [m] maximum foot swing height during walking

    # Foot geometry for stability analysis
    FOOT_LENGTH: float = 0.16  # [m] foot length (anterior-posterior)
    FOOT_WIDTH: float = 0.08  # [m] foot width (medial-lateral)

    # ZMP sway default parameters
    SWAY_AMPLITUDE: float = 0.02  # [m] default lateral sway amplitude
    SWAY_FREQUENCY: float = 0.2  # [Hz] default sway frequency
    SWAY_RAMP_START: float = 2.0  # [s] time before sway begins
    SWAY_RAMP_DURATION: float = 3.0  # [s] time to reach full amplitude

    # LQR internal parameters (computed automatically)
    N_PREVIEW: int = None  # Computed: number of preview samples (PREVIEW_TIME / DT)

    def __post_init__(self):
        # Calculate preview samples for discrete implementation
        if self.DT > 0:
            self.N_PREVIEW = int(self.PREVIEW_TIME / self.DT)


@dataclass
class PhaseManagerConfig:
    """Walking phase manager parameters for G1 robot locomotion."""
    # Phase timing (seconds)
    SETTLE_TIME: float = 7.0  # Time to settle robot on ground after initialization
    BALANCE_TIME: float = 15.0  # Time to maintain static balance before starting walk
    PREPARE_TIME: float = 0.3  # Time to prepare for next step (transition to single support)
    STEP_TIME: float = 0.6  # Duration of single support phase (stepping)
    DOUBLE_SUPPORT_TIME: float = 0.2  # Time in double support phase (both feet on ground)
    STABILITY_HOLD_TIME: float = 5.0  # Time to hold and verify stability
    ZMP_SWAY_DURATION: float = 40.0  # Duration of ZMP sway phase

    # Stability margins for COM (Center of Mass) control
    COM_X_MARGIN: float = 0.1  # [m] safety margin for COM X position relative to ZMP
    COM_Y_MARGIN: float = 0.05  # [m] safety margin for COM Y position relative to ZMP

    # Foot placement parameters
    FOOT_SEPARATION: float = 0.15  # [m] initial distance between left and right feet
    STEP_OFFSET: float = 0.1  # [m] offset for foot placement from center line

    # Balance ramp parameters
    BALANCE_RAMP_RATIO: float = 0.5  # Fraction of balance time for ramping (rest is holding)

    # Real-time pacing multipliers
    SETTLE_PACE: float = 1.0  # Multiplier for settle phase (< 1 = faster than real-time)
    BALANCE_PACE: float = 1.0  # Multiplier for balance phase
    HOLD_PACE: float = 1.0  # Multiplier for stability hold phase

    # Status reporting interval
    STATUS_INTERVAL: float = 0.5  # [s] How often to print status during phases


@dataclass
class WholeBodyIKConfig:
    """Whole body inverse kinematics parameters for G1 robot."""
    # Task-space proportional gains (for velocity computation from error)
    KP_FOOT_POSITION: float = 200.0  # [1/s] Foot position tracking gain
    # High because feet MUST stay planted (ground contact)
    # 200 means: 1cm error → 2 m/s desired correction velocity
    KP_FOOT_ROTATION: float = 100.0  # [1/s] Foot orientation tracking gain
    # High but less than position — orientation drift is less critical
    KP_COM: float = 20.0  # [1/s] CoM tracking gain
    # Lower than feet — CoM can move more slowly

    # Task weights for weighted least-squares optimization
    WEIGHT_FOOT_POSITION: float = 2000.0  # Weight for foot position tracking error
    WEIGHT_FOOT_ROTATION: float = 500.0  # Weight for foot orientation tracking error
    WEIGHT_COM: float = 100.0  # Weight for CoM tracking

    # Regularization parameters
    JOINT_REG_WEIGHT: float = 2.0  # Weight for joint regularization (keeping joints near nominal)
    DAMPING: float = 1e-4  # Tikhonov damping for pseudo-inverse stability

    # Velocity limits
    MAX_JOINT_VELOCITY: float = 2.0  # [rad/s] Maximum joint velocity in IK solution
    # 4 rad/s ≈ 230 deg/s — fast but physically plausible for a humanoid

    # Optimization parameters
    MAX_ITERATIONS: int = 100  # Maximum iterations for IK optimization
    DEFAULT_ITERATIONS: int = 10  # Default iterations per solve call
    TRACKING_ITERATIONS: int = 5  # Iterations when in tracking mode (synced every tick)
    TOLERANCE: float = 1e-6  # Convergence tolerance for optimization

    # Joint limits safety margin
    JOINT_LIMIT_MARGIN: float = 0.05  # [rad] safety margin from joint limits


@dataclass
class VisualizationConfig:
    """Simulation visualization and debugging parameters."""
    # Visualization toggles
    SHOW_COORDINATES: bool = True  # Show coordinate frames at body joints
    SHOW_CONTACT_POINTS: bool = True  # Show contact points between robot and ground
    SHOW_COM_TRAJECTORY: bool = True  # Show Center of Mass trajectory
    SHOW_ZMP_TRAJECTORY: bool = True  # Show Zero Moment Point trajectory

    # Camera settings for MuJoCo viewer
    CAMERA_DISTANCE: float = 2.5  # [m] distance from robot to camera
    CAMERA_AZIMUTH: float = 135.0  # [deg] horizontal rotation around lookat
    CAMERA_ELEVATION: float = -20.0  # [deg] camera angle above horizontal (negative = looking down)
    CAMERA_LOOKAT: Tuple[float, float, float] = (0.0, 0.0, 0.7)  # [m] point camera centers on

    # Timing
    INITIAL_PAUSE: float = 1.0  # [s] pause after viewer launch before simulation starts
    IDLE_SLEEP: float = 0.01  # [s] sleep time when viewer is idle


@dataclass
class ControllerConfig:
    """Robot controller configuration parameters."""
    CONTROL_MODE: ControlMode = ControlMode.POSITION  # Position or torque control
    ENABLE_PREVIEW: bool = True  # Enable ZMP preview control

    # Robot-specific controller settings
    Kp_LIST: Optional[Dict[str, float]] = None  # Per-joint proportional gains (position mode)
    Kd_LIST: Optional[Dict[str, float]] = None  # Per-joint derivative gains (position mode)
    Torque_LIMIT: float = 150.0  # [Nm] Maximum torque per joint (torque mode)

    # IK sync frequency
    IK_SYNC_INTERVAL: int = 2  # Sync IK with sim every N ticks during ZMP control

    def __post_init__(self):
        if self.Kp_LIST is None:
            # Default per-joint gains based on robot_model.py analysis
            # Hip/Knee: high stiffness, Ankle: medium, Waist: medium, Arms: low
            self.Kp_LIST = {
                "hip": 400.0,  # High gains for large masses (hip/knee)
                "knee": 400.0,
                "ankle": 300.0,  # Medium gains for balance
                "waist": 200.0,  # Medium gains for upper body stability
                "arm": 80.0,  # Low gains for fine control
                "default": 200.0  # Default fallback
            }
        if self.Kd_LIST is None:
            self.Kd_LIST = {
                "hip": 40.0,
                "knee": 40.0,
                "ankle": 30.0,
                "waist": 20.0,
                "arm": 8.0,
                "default": 20.0
            }


# ================================================================
# MAIN CONFIGURATION CLASS
# ================================================================
@dataclass
class Config:
    """Main configuration container for ZMP Humanoid Project."""

    # Configuration sections
    SIMULATION: SimulationConfig = field(default_factory=SimulationConfig)
    ROBOT: RobotModelConfig = field(default_factory=RobotModelConfig)
    ZMP: ZMPControllerConfig = field(default_factory=ZMPControllerConfig)
    PHASE: PhaseManagerConfig = field(default_factory=PhaseManagerConfig)
    IK: WholeBodyIKConfig = field(default_factory=WholeBodyIKConfig)
    VISUALIZATION: VisualizationConfig = field(default_factory=VisualizationConfig)
    CONTROLLER: ControllerConfig = field(default_factory=ControllerConfig)

    # Debug and logging
    DEBUG_MODE: bool = False
    VERBOSE: bool = True
    LOG_DIR: str = "logs"

    def __post_init__(self):
        """Validate configuration and compute derived values."""
        # Ensure control timestep is multiple of simulation timestep
        if self.SIMULATION.DT > self.SIMULATION.CONTROL_DT:
            raise ValueError("CONTROL_DT must be >= SIMULATION.DT")

        # Validate ZMP parameters
        if self.ZMP.Z_C <= 0:
            raise ValueError("Z_C (CoM height) must be positive")

        # Validate robot model
        if self.ROBOT.KP_DEFAULT <= 0 or self.ROBOT.KD_DEFAULT <= 0:
            raise ValueError("Joint gains must be positive")

        # Validate phase times sum to reasonable values
        total_cycle = (self.PHASE.SETTLE_TIME + self.PHASE.BALANCE_TIME +
                       self.PHASE.STEP_TIME + self.PHASE.DOUBLE_SUPPORT_TIME)
        if total_cycle < 2.0:
            print("Warning: Total phase times are less than 2 seconds")

    # Helper methods for common calculations
    def get_controller_dt(self) -> float:
        """Get effective controller timestep."""
        return self.SIMULATION.CONTROL_DT

    def get_steps_per_control(self) -> int:
        """Get number of physics steps per control step."""
        return self.SIMULATION.STEPS_PER_CONTROL

    def get_preview_samples(self) -> int:
        """Get number of preview samples for ZMP controller."""
        return self.ZMP.N_PREVIEW

    def get_control_frequency(self) -> float:
        """Get control loop frequency in Hz."""
        return 1.0 / self.SIMULATION.CONTROL_DT

    def get_physics_frequency(self) -> float:
        """Get physics simulation frequency in Hz."""
        return 1.0 / self.SIMULATION.DT

    def get_gravity_magnitude(self) -> float:
        """Get magnitude of gravity (positive value)."""
        return abs(self.SIMULATION.GRAVITY[2])


# ================================================================
# INSTANCE
# ================================================================
# Create a global instance of the configuration
cfg = Config()


# Helper function to get configuration
def get_config() -> Config:
    """Get the global configuration instance."""
    return cfg
