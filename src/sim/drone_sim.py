"""PyBullet-based quadcopter simulation with motor dynamics."""

from __future__ import annotations

import math
import random
from typing import List

import numpy as np
import pybullet as p
import pybullet_data


class DroneSim:
    """
    Minimal PyBullet simulation for a quadcopter.

    Motor dynamics:
        1. First-order lag:  motor_state += (cmd - motor_state) * (dt / tau)
        2. Exponential smoothing: motor_smooth = alpha * motor_prev + (1 - alpha) * motor_state
    """

    # Motor link indices inside the URDF (motor1..motor4 are children of base_link)
    MOTOR_LINK_INDICES = [0, 1, 2, 3]

    def __init__(
        self,
        urdf_path: str = "assets/urdf/quad.urdf",
        gui: bool = True,
        time_step: float = 1.0 / 240.0,
        max_thrust_per_motor: float = 10.0,
        motor_tau: float = 0.10,
        motor_alpha: float = 0.9,
    ) -> None:
        """
        Initialise the simulation environment.

        Args:
            urdf_path: Path to the quadcopter URDF file.
            gui: If True, open a GUI window; otherwise run headless.
            time_step: Physics time step in seconds (default 240 Hz).
            max_thrust_per_motor: Maximum thrust (N) when throttle=1.
            motor_tau: Time constant for first-order motor lag (seconds).
            motor_alpha: Smoothing factor for exponential moving average (0-1).
        """
        self.urdf_path = urdf_path
        self.time_step = time_step
        self.max_thrust = max_thrust_per_motor
        self.motor_tau = motor_tau
        self.motor_alpha = motor_alpha

        # Motor internal states (both in [0, 1])
        self._motor_state = np.zeros(4, dtype=np.float32)
        self._motor_smooth = np.zeros(4, dtype=np.float32)

        # Connect to physics server
        mode = p.GUI if gui else p.DIRECT
        self.physics_client = p.connect(mode)

        # Configure simulation
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.time_step)

        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")

        # Load quadcopter
        self.drone_id: int = -1
        self._spawn_drone()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> List[float]:
        """
        Reset the drone to starting position with small random orientation.

        Returns:
            Initial sensor readings after reset.
        """
        # Remove existing drone and respawn
        if self.drone_id >= 0:
            p.removeBody(self.drone_id)

        # Reset motor dynamics
        self._motor_state[:] = 0.0
        self._motor_smooth[:] = 0.0

        self._spawn_drone()
        return self.get_sensor_readings()

    def set_motor_commands(self, commands: np.ndarray) -> None:
        """
        Set desired motor commands (throttles in [0, 1]).

        The actual thrust will lag behind due to motor dynamics.
        Call step() afterwards to advance physics and apply forces.
        """
        commands = np.asarray(commands, dtype=np.float32)
        commands = np.clip(commands, 0.0, 1.0)

        # First-order lag update
        dt_over_tau = self.time_step / self.motor_tau
        self._motor_state += (commands - self._motor_state) * dt_over_tau

        # Exponential smoothing
        self._motor_smooth = (
            self.motor_alpha * self._motor_smooth
            + (1.0 - self.motor_alpha) * self._motor_state
        )

    def apply_motor_forces(self, throttles: List[float] | np.ndarray | None = None) -> None:
        """
        Apply upward thrust at each motor mount point.

        If throttles is None, uses internal smoothed motor state.
        Otherwise throttles are applied directly (legacy behaviour).
        """
        if throttles is None:
            effective = self._motor_smooth
        else:
            effective = np.clip(np.asarray(throttles, dtype=np.float32), 0.0, 1.0)

        for idx in range(4):
            force_magnitude = float(effective[idx]) * self.max_thrust
            force_vector = [0.0, 0.0, force_magnitude]
            p.applyExternalForce(
                objectUniqueId=self.drone_id,
                linkIndex=self.MOTOR_LINK_INDICES[idx],
                forceObj=force_vector,
                posObj=[0.0, 0.0, 0.0],
                flags=p.WORLD_FRAME,
            )

    def step(self) -> None:
        """Advance the simulation by one time step."""
        p.stepSimulation()

    def get_sensor_readings(self) -> List[float]:
        """
        Retrieve current sensor readings.

        Returns:
            [roll, pitch, yaw, vx, vy, vz, gx, gy, gz]
            Angles in radians, velocities in m/s and rad/s.
        """
        pos, orn = p.getBasePositionAndOrientation(self.drone_id)
        roll, pitch, yaw = p.getEulerFromQuaternion(orn)

        lin_vel, ang_vel = p.getBaseVelocity(self.drone_id)
        vx, vy, vz = lin_vel
        gx, gy, gz = ang_vel

        return [roll, pitch, yaw, vx, vy, vz, gx, gy, gz]

    def get_position(self) -> np.ndarray:
        """Return drone (x, y, z) position."""
        pos, _ = p.getBasePositionAndOrientation(self.drone_id)
        return np.array(pos, dtype=np.float32)

    def disconnect(self) -> None:
        """Disconnect from the physics server."""
        p.disconnect(self.physics_client)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _spawn_drone(self) -> None:
        """Load the drone URDF with a small random roll/pitch perturbation."""
        start_pos = [0.0, 0.0, 1.0]

        roll = random.uniform(-5, 5) * math.pi / 180.0
        pitch = random.uniform(-5, 5) * math.pi / 180.0
        yaw = 0.0
        start_orn = p.getQuaternionFromEuler([roll, pitch, yaw])

        self.drone_id = p.loadURDF(
            self.urdf_path,
            basePosition=start_pos,
            baseOrientation=start_orn,
            useFixedBase=False,
        )


# ------------------------------------------------------------------
# Quick manual test
# ------------------------------------------------------------------
if __name__ == "__main__":
    import time

    sim = DroneSim(gui=True, max_thrust_per_motor=10.0)
    print("Sensor readings after spawn:", sim.get_sensor_readings())

    hover_throttle = np.array([0.3, 0.3, 0.3, 0.3], dtype=np.float32)

    for _ in range(2400):  # 10 seconds at 240 Hz
        sim.set_motor_commands(hover_throttle)
        sim.apply_motor_forces()
        sim.step()
        time.sleep(sim.time_step)

    print("Sensor readings after 10s:", sim.get_sensor_readings())
    input("Press ENTER to close...")
    sim.disconnect()
