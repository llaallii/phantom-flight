# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Phantom-Flight is a physics-accurate quadcopter digital twin with RL training pipeline. It combines PyBullet simulation with Gymnasium-compatible RL environments for training deployable flight controllers.

**Philosophy:** Physics first, intelligence later. The simulation prioritizes realistic motor dynamics over convenience.

## Development Commands

### Environment Setup
```bash
conda create -n phantom-flight python=3.11 -y
conda activate phantom-flight
pip install pybullet gymnasium numpy stable-baselines3 torch onnx onnxruntime
```

Or: `pip install -r requirements.txt`

Verify: `python -c "import pybullet; import stable_baselines3; print('OK')"`

### Running the Simulator
```bash
# Interactive keyboard control demo
python -m src.tools.play_sim
```

### Training RL Models
```bash
# Train with SAC (recommended)
python -m src.train.train --algo sac --timesteps 500000

# Train with TD3
python -m src.train.train --algo td3 --timesteps 500000

# Train with GUI (slow, for debugging)
python -m src.train.train --algo sac --timesteps 100000 --gui
```

### Training with Hindsight Experience Replay (HER)
```bash
# Train with SAC + HER (sparse rewards)
python -m src.train.train --algo sac --timesteps 500000 --her

# Train with HER using dense rewards
python -m src.train.train --algo sac --timesteps 500000 --her --reward-type dense

# Train with HER and randomized goals
python -m src.train.train --algo sac --timesteps 500000 --her --randomize-goal

# Train with different HER strategy (future, final, or episode)
python -m src.train.train --algo sac --timesteps 500000 --her --her-strategy final

# Full HER configuration example
python -m src.train.train --algo sac --timesteps 500000 --her \
    --her-strategy future --her-n-goals 4 \
    --reward-type sparse --distance-threshold 0.1 --randomize-goal
```

HER is only compatible with off-policy algorithms (SAC, TD3, DDPG), not PPO.

Training outputs:
- Models saved to `outputs/models/`
- TensorBoard logs saved to `outputs/logs/`

## Architecture

### Core Components

1. **DroneSim** (`src/sim/drone_sim.py`)
   - PyBullet-based physics simulation running at 240 Hz
   - Implements realistic motor dynamics with first-order lag and exponential smoothing
   - Motor dynamics: `motor_state += (cmd - motor_state) * (dt / tau)` followed by smoothing
   - Default parameters: `tau=0.10s`, `alpha=0.9`
   - URDF location: `assets/urdf/simple_quad.urdf` (1 kg quadcopter, X-config)
   - Motor link indices: [0, 1, 2, 3] corresponding to 4 motor mounts

2. **QuadHoverEnv** (`src/env/hil_env.py`)
   - Gymnasium-compatible environment for continuous control RL
   - Control rate: 40 Hz (6 sub-steps per action at 240 Hz physics)
   - **Action space:** Box([-1, 1], shape=(4,)) - mapped to motor throttles via `(action + 1) / 2`
   - **Observation space:** 12-dim vector: `[roll, pitch, yaw, vx, vy, vz, gx, gy, gz, altitude_error, x, y]`
   - **Termination conditions:** altitude < 0.1m, tilt > 60°, or NaN values

3. **QuadHoverGoalEnv** (`src/env/hil_env.py`)
   - Goal-conditioned environment for Hindsight Experience Replay (HER)
   - Uses Dict observation space required by HER:
     - `observation`: 9-dim `[roll, pitch, yaw, vx, vy, vz, gx, gy, gz]`
     - `achieved_goal`: 3-dim `[x, y, z]` - current position
     - `desired_goal`: 3-dim `[x, y, z]` - target position
   - **Reward types:**
     - Sparse: -1 if distance > threshold, 0 otherwise
     - Dense: negative Euclidean distance to goal
   - **Goal randomization:** Optional random goal positions within bounds
   - Uses `compute_reward(achieved_goal, desired_goal, info)` interface for HER relabeling

4. **Training Pipeline** (`src/train/train.py`)
   - Supports TD3, SAC, DDPG, PPO via Stable-Baselines3
   - Policy: Small MLP [64, 64] designed for embedded deployment (STM32 target)
   - Evaluation and checkpointing callbacks included
   - All training environments are vectorized via DummyVecEnv

### Motor Configuration (X-configuration)
- Motor 1: Front-right
- Motor 2: Front-left
- Motor 3: Rear-left
- Motor 4: Rear-right

### Reward Function (per step, O(1) scale)
The reward function balances multiple objectives:
- Altitude error: `-2.0 * (z - target)^2`
- Tilt penalty: `-1.0 * (roll^2 + pitch^2)`
- Angular velocity: `-0.1 * (gx^2 + gy^2 + gz^2)`
- Linear velocity: `-0.1 * (vx^2 + vy^2 + vz^2)`
- Position drift: `-0.2 * (x^2 + y^2)`
- Action smoothness: `-0.05 * sum(action^2)`
- Alive bonus: `+0.1`

## Key Design Decisions

### Motor Dynamics
Real quadcopters cannot instantaneously change thrust. Motor dynamics are modeled with:
1. First-order lag (tau ≈ 0.10s) prevents instant response
2. Exponential smoothing (alpha = 0.9) adds realistic inertia

This makes the control problem realistic and prevents "magic" instantaneous control.

### Control Frequency
- **Physics rate:** 240 Hz (required for stable simulation)
- **Control rate:** 40 Hz (6 sub-steps between actions)
- This mimics real embedded systems where control loops run at lower frequencies than sensor sampling

### Small Policy Networks
Policy uses MLP [64, 64] to ensure deployment feasibility on STM32 microcontrollers. Avoid adding layers or complexity without considering embedded constraints.

### Hindsight Experience Replay (HER)
HER improves sample efficiency for goal-conditioned tasks by relabeling failed trajectories with achieved goals. This helps the agent learn from failures.

**Goal Selection Strategies:**
- `future`: Sample goals from states visited later in the same episode (recommended, default)
- `final`: Use the final state of the episode as the goal
- `episode`: Sample goals uniformly from states in the same episode

**When to Use HER:**
- Sparse reward settings where success is rare
- Position/waypoint reaching tasks
- When randomizing goals across episodes
- When standard RL struggles to find any successful trajectories

**Parameters:**
- `--distance-threshold`: How close to goal counts as success (default: 0.1m)
- `--her-n-goals`: Virtual goals per transition (default: 4, higher = more relabeling)
- `--reward-type`: `sparse` (-1/0) for pure HER, `dense` for shaped rewards

## Development Guidelines

### When Modifying Simulation Physics
- Always maintain deterministic behavior (no uncontrolled randomness in step())
- Respect the 240 Hz physics timestep - changing it affects motor dynamics
- Test with manual keyboard control (`play_sim.py`) before RL training

### When Modifying the Environment
- Keep observation space bounded and normalized
- Ensure reward components are O(1) per step for training stability
- Test termination conditions don't trigger prematurely
- Random initial tilt (±5°) in reset() provides exploration diversity

### When Training Models
- SAC generally outperforms other algorithms for this task
- Headless training (no --gui) is 10-20x faster
- Evaluation environment must be separate from training environment
- Monitor TensorBoard logs for reward progression and episode length

### URDF Modifications
The quadcopter URDF (`assets/urdf/simple_quad.urdf`) defines:
- Mass: 1 kg total
- Motor positions in X-configuration
- Collision geometry
- Inertial properties

Changes to mass/geometry require retuning `max_thrust_per_motor` parameter.

## System Requirements
- Python 3.10+ (tested on 3.11)
- Conda recommended for environment management
- OpenGL-capable GPU for PyBullet GUI rendering
- Windows/Linux/macOS supported

## Future Roadmap
- ONNX export of trained policies
- STM32 deployment scripts
- Hardware-in-the-loop (HIL) testing integration
