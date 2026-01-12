---

# Phantom-Flight

### Physics-Accurate Quadcopter Digital Twin + RL Training Pipeline

**Phantom-Flight** is a software-first, physics-driven quadcopter simulation platform designed to produce *deployable flight brains*.

The repository now contains:

1. **Simulation Core** – a deterministic PyBullet digital twin with realistic motor dynamics.
2. **Gymnasium RL Environment** – continuous-control hover task compatible with TD3/SAC/DDPG/PPO.
3. **SB3 Training Script** – ready-to-run training with small MLP policies suitable for embedded deployment.

---

## What Exists in This Repo

• PyBullet physics world (240 Hz)
• Rigid-body quadcopter with 4 motor mounts (X-config)
• First-order motor lag + exponential smoothing
• Gymnasium-compatible `QuadHoverEnv`
• Stable-Baselines3 training script (`train_ppo.py`)
• Interactive keyboard demo (`play_sim.py`)

---

## Repository Layout

```
phantom-flight/
├── assets/urdf/
│   └── simple_quad.urdf        # 1 kg quadcopter URDF
├── outputs/
│   ├── logs/                   # Tensorboard logs
│   └── models/                 # Saved models & checkpoints
├── src/
│   ├── env/
│   │   └── hil_env.py          # QuadHoverEnv (Gymnasium)
│   ├── sim/
│   │   └── drone_sim.py        # DroneSim with motor dynamics
│   ├── tools/
│   │   └── play_sim.py         # Manual keyboard control
│   └── train/
│       └── train.py            # SB3 training (TD3/SAC/DDPG/PPO)
├── requirements.txt
└── README.md
```

---

## System Requirements

• Python 3.10+ (tested on 3.11)
• Conda recommended
• OpenGL-capable GPU for PyBullet GUI

---

## Installation

```bash
conda create -n phantom-flight python=3.11 -y
conda activate phantom-flight
pip install pybullet gymnasium numpy stable-baselines3 torch onnx onnxruntime
```

Or from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Verify:

```bash
python -c "import pybullet; import stable_baselines3; print('OK')"
```

---

## Running the Simulator (Manual Control)

```bash
python -m src.tools.play_sim
```

| Key | Action |
|-----|--------|
| W / S | Base throttle ↑ / ↓ |
| I / K | Trim motor 1 |
| J / L | Trim motor 2 |
| U / O | Trim motor 3 |
| Y / H | Trim motor 4 |
| R | Reset |
| Q / Esc | Quit |

---

## Motor Dynamics Model

```
motor_state += (command - motor_state) * (dt / tau)
motor_smooth = alpha * motor_prev + (1 - alpha) * motor_state
```

• `tau` ≈ 0.10 s (first-order lag)
• `alpha` = 0.9 (exponential smoothing)

This prevents instantaneous thrust changes and makes the control problem realistic.

---

## RL Environment (`QuadHoverEnv`)

| Property | Value |
|----------|-------|
| Action space | `Box([-1,1], shape=(4,))` |
| Observation | 12-dim: roll, pitch, yaw, vx, vy, vz, gx, gy, gz, alt_error, x, y |
| Physics rate | 240 Hz |
| Control rate | 40 Hz (6 sub-steps per action) |
| Termination | altitude < 0.1 m, tilt > 60°, NaN |

### Action Mapping

```python
throttle = clip((action + 1) / 2, 0, 1)
```

### Reward (per step, O(1) scale)

```
r_alt   = -2.0 * (z - target)^2
r_tilt  = -1.0 * (roll^2 + pitch^2)
r_ang   = -0.1 * (gx^2 + gy^2 + gz^2)
r_vel   = -0.1 * (vx^2 + vy^2 + vz^2)
r_pos   = -0.2 * (x^2 + y^2)
r_act   = -0.05 * sum(action^2)
alive   = +0.1
```

---

## Training

```bash
# SAC (recommended)
python -m src.train.train --algo sac --timesteps 500000

# TD3
python -m src.train.train --algo td3 --timesteps 500000

# With GUI (slow)
python -m src.train.train --algo sac --timesteps 100000 --gui
```

Models and TensorBoard logs are saved to `outputs/models/` and `outputs/logs/`.

Policy architecture: MLP `[64, 64]` (suitable for STM32 deployment).

---

## DroneSim API

| Method | Purpose |
|--------|---------|
| `reset()` | Respawn drone with random ±5° tilt |
| `set_motor_commands(cmd)` | Update motor state with lag/smoothing |
| `apply_motor_forces()` | Apply smoothed thrust to motors |
| `step()` | Advance physics (240 Hz) |
| `get_sensor_readings()` | `[roll, pitch, yaw, vx, vy, vz, gx, gy, gz]` |
| `get_position()` | `[x, y, z]` |

---

## Roadmap

- [x] PyBullet simulation with URDF
- [x] Motor dynamics (lag + smoothing)
- [x] Gymnasium-compatible `QuadHoverEnv`
- [x] SB3 training script (TD3/SAC/DDPG/PPO)
- [ ] ONNX export of trained policy
- [ ] STM32 deployment & HIL testing

---

## Project Philosophy

> **Physics first. Intelligence later.**
> Reality does not bend to neural networks — neural networks must bend to reality.

---

