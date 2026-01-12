"""Interactive keyboard-controlled quadcopter demo using DroneSim."""

from __future__ import annotations

import sys
import time

import pybullet as p

# Ensure src package is importable when running as script
sys.path.insert(0, ".")

from src.sim.drone_sim import DroneSim


def play() -> None:
    """
    Run the drone simulation with keyboard control.

    Controls (press & release in PyBullet GUI window):
        W / S   – increase / decrease base throttle (all motors)
        I       – trim motor 1 (front-right) up
        K       – trim motor 1 down
        J       – trim motor 2 (front-left) up
        L       – trim motor 2 down
        U       – trim motor 3 (rear-left) up
        O       – trim motor 3 down
        Y       – trim motor 4 (rear-right) up
        H       – trim motor 4 down
        R       – reset drone
        Q / ESC – quit
    """

    sim = DroneSim(gui=True, max_thrust_per_motor=12.0)

    base_throttle = 0.0
    trim = [0.0, 0.0, 0.0, 0.0]  # per-motor adjustments

    throttle_step = 0.02
    trim_step = 0.01

    print(__doc__)
    print("Starting interactive control. Focus the PyBullet window and use keys.")

    step_count = 0
    running = True

    while running:
        # ---- Read keyboard events from PyBullet ----
        keys = p.getKeyboardEvents()

        for key, state in keys.items():
            if state & p.KEY_WAS_TRIGGERED or state & p.KEY_IS_DOWN:
                # Base throttle
                if key == ord("w"):
                    base_throttle = min(1.0, base_throttle + throttle_step)
                elif key == ord("s"):
                    base_throttle = max(0.0, base_throttle - throttle_step)

                # Per-motor trims
                elif key == ord("i"):
                    trim[0] = min(0.3, trim[0] + trim_step)
                elif key == ord("k"):
                    trim[0] = max(-0.3, trim[0] - trim_step)
                elif key == ord("j"):
                    trim[1] = min(0.3, trim[1] + trim_step)
                elif key == ord("l"):
                    trim[1] = max(-0.3, trim[1] - trim_step)
                elif key == ord("u"):
                    trim[2] = min(0.3, trim[2] + trim_step)
                elif key == ord("o"):
                    trim[2] = max(-0.3, trim[2] - trim_step)
                elif key == ord("y"):
                    trim[3] = min(0.3, trim[3] + trim_step)
                elif key == ord("h"):
                    trim[3] = max(-0.3, trim[3] - trim_step)

                # Reset
                elif key == ord("r"):
                    sim.reset()
                    base_throttle = 0.0
                    trim = [0.0, 0.0, 0.0, 0.0]
                    print(">> Drone reset")

                # Quit (q or Escape – key code 27)
                elif key == ord("q") or key == 27:
                    running = False

        # ---- Build throttle vector and apply ----
        throttles = [max(0.0, min(1.0, base_throttle + t)) for t in trim]
        sim.apply_motor_forces(throttles)
        sim.step()

        # ---- Periodic status print ----
        if step_count % 480 == 0:  # every 2 seconds at 240 Hz
            readings = sim.get_sensor_readings()
            print(
                f"Throttles: {[round(t, 2) for t in throttles]}  |  "
                f"Roll={readings[0]:.2f} Pitch={readings[1]:.2f} Yaw={readings[2]:.2f}  |  "
                f"Z-vel={readings[5]:.2f}"
            )

        step_count += 1
        time.sleep(sim.time_step)

    sim.disconnect()
    print("Simulation ended.")


if __name__ == "__main__":
    play()
