"""
Gymnasium-compatible quadcopter hover environment.

Designed for continuous-control actor–critic algorithms (TD3, DDPG, SAC).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.sim.drone_sim import DroneSim


class QuadHoverEnv(gym.Env):
    """
    Continuous-control hover environment for a quadcopter.

    Observation (12-dim):
        [roll, pitch, yaw, vx, vy, vz, gx, gy, gz, z, prev_action (4)]
        → actually 9 + 1 + 4 = 14 but we keep simpler: 12 dims for now
        Simplified: [roll, pitch, yaw, vx, vy, vz, gx, gy, gz, z-1, dx, dy]
        Final: [roll, pitch, yaw, vx, vy, vz, gx, gy, gz, altitude_error, x, y]

    Action (4-dim, continuous):
        Each element in [-1, 1].
        Internally mapped to motor throttle [0, 1] via: throttle = clip((a+1)/2, 0, 1).

    Reward:
        Designed so each component is O(1) per timestep.

    Termination:
        - altitude < 0.1 m
        - roll or pitch > 60°
        - NaN / runaway
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    # Limits (radians, m/s)
    MAX_TILT = math.radians(60)
    MIN_ALT = 0.1
    MAX_ALT = 5.0
    MAX_VEL = 10.0
    MAX_ANG_VEL = 10.0

    def __init__(
        self,
        gui: bool = False,
        max_episode_steps: int = 1000,
        control_freq: float = 40.0,
        physics_freq: float = 240.0,
        max_thrust_per_motor: float = 10.0,
        motor_tau: float = 0.10,
        motor_alpha: float = 0.9,
        target_altitude: float = 1.0,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.gui = gui
        self.max_episode_steps = max_episode_steps
        self.control_freq = control_freq
        self.physics_freq = physics_freq
        self.sub_steps = int(physics_freq / control_freq)
        self.target_altitude = target_altitude
        self.render_mode = render_mode

        self._sim_params = dict(
            gui=gui,
            time_step=1.0 / physics_freq,
            max_thrust_per_motor=max_thrust_per_motor,
            motor_tau=motor_tau,
            motor_alpha=motor_alpha,
        )
        self.sim: Optional[DroneSim] = None

        # Action: 4 motors, each in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # Observation: 12 floats (see _get_obs)
        obs_high = np.array(
            [
                np.pi,          # roll
                np.pi,          # pitch
                np.pi,          # yaw
                self.MAX_VEL,   # vx
                self.MAX_VEL,   # vy
                self.MAX_VEL,   # vz
                self.MAX_ANG_VEL,  # gx
                self.MAX_ANG_VEL,  # gy
                self.MAX_ANG_VEL,  # gz
                self.MAX_ALT,   # altitude error (signed)
                5.0,            # x drift
                5.0,            # y drift
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-obs_high, high=obs_high, dtype=np.float32
        )

        self._step_count = 0
        self._prev_action = np.zeros(4, dtype=np.float32)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        if self.sim is None:
            self.sim = DroneSim(**self._sim_params)
        self.sim.reset()

        self._step_count = 0
        self._prev_action = np.zeros(4, dtype=np.float32)

        obs = self._get_obs()
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)

        # Map action [-1, 1] → throttle [0, 1]
        throttle = np.clip((action + 1.0) / 2.0, 0.0, 1.0)

        # Run multiple physics sub-steps per control step
        for _ in range(self.sub_steps):
            self.sim.set_motor_commands(throttle)
            self.sim.apply_motor_forces()
            self.sim.step()

        self._step_count += 1
        self._prev_action = action.copy()

        obs = self._get_obs()
        reward = self._compute_reward(obs, action)
        terminated = self._check_termination(obs)
        truncated = self._step_count >= self.max_episode_steps

        info: Dict[str, Any] = {}
        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        # GUI mode handled by PyBullet; nothing extra needed
        return None

    def close(self) -> None:
        if self.sim is not None:
            self.sim.disconnect()
            self.sim = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        readings = self.sim.get_sensor_readings()
        pos = self.sim.get_position()

        roll, pitch, yaw = readings[0], readings[1], readings[2]
        vx, vy, vz = readings[3], readings[4], readings[5]
        gx, gy, gz = readings[6], readings[7], readings[8]
        z = pos[2]
        x, y = pos[0], pos[1]

        alt_error = z - self.target_altitude

        obs = np.array(
            [roll, pitch, yaw, vx, vy, vz, gx, gy, gz, alt_error, x, y],
            dtype=np.float32,
        )

        # Clip to observation space bounds (safety)
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        return obs

    def _compute_reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        """
        Reward designed so each term is O(1) per step.
        """
        roll, pitch, yaw = obs[0], obs[1], obs[2]
        vx, vy, vz = obs[3], obs[4], obs[5]
        gx, gy, gz = obs[6], obs[7], obs[8]
        alt_error = obs[9]
        x, y = obs[10], obs[11]

        # Altitude penalty (quadratic, scaled)
        r_alt = -2.0 * (alt_error ** 2)

        # Tilt penalty
        r_tilt = -1.0 * (roll ** 2 + pitch ** 2)

        # Angular velocity penalty
        r_ang = -0.1 * (gx ** 2 + gy ** 2 + gz ** 2)

        # Linear velocity penalty (horizontal drift and vertical speed)
        r_vel = -0.1 * (vx ** 2 + vy ** 2 + vz ** 2)

        # Position drift penalty (encourage staying at origin)
        r_pos = -0.2 * (x ** 2 + y ** 2)

        # Action smoothness (penalise large actions)
        r_act = -0.05 * float(np.sum(action ** 2))

        reward = r_alt + r_tilt + r_ang + r_vel + r_pos + r_act

        # Small alive bonus
        reward += 0.1

        return float(reward)

    def _check_termination(self, obs: np.ndarray) -> bool:
        roll, pitch = obs[0], obs[1]
        alt_error = obs[9]
        z = alt_error + self.target_altitude

        # Altitude too low
        if z < self.MIN_ALT:
            return True

        # Excessive tilt
        if abs(roll) > self.MAX_TILT or abs(pitch) > self.MAX_TILT:
            return True

        # NaN check
        if np.any(np.isnan(obs)):
            return True

        return False


# Alias for convenience / backward compatibility
HardwareInLoopEnv = QuadHoverEnv
