# Quadruped RL Locomotion: Unitree Go1 with PPO in Isaac Lab

This repository contains the code and results for the final project of **ECE 489 / ME 446 (Spring 2026)** at Zhejiang University, implementing a learning‑based locomotion controller for the Unitree Go1 quadruped robot using **Proximal Policy Optimization (PPO)** in NVIDIA Isaac Lab.

## Project Overview

We train two policies to trot stably:
- **Flat terrain** – simple plane with friction 0.8
- **Rough terrain** – slopes (~12°), random boxes (2.5–10 cm), and Perlin noise (1–6 cm)

The policy uses proprioceptive observations (joint states, velocity, gravity, commands) plus a 187‑D height scan on rough terrain. Domain randomization (friction, payload mass, motor strength) improves robustness. The PPO implementation follows RSL‑RL with separate actor‑critic networks.

## Key Results

| Metric                         | Flat        | Rough       |
|--------------------------------|-------------|-------------|
| Velocity tracking error (RMS)  | 0.11 m/s    | 0.18 m/s    |
| Body roll RMS                  | 2.4°        | 4.1°        |
| Body pitch RMS                 | 1.9°        | 3.8°        |
| Success rate (10 s)            | 90 %        | 80 %        |
| Cost of Transport              | 0.38        | 0.52        |
| Push recovery                  | 100 %       | 70 %        |

Training details:
- Flat policy: 800 iterations (19.66 M steps)
- Rough policy: 1500 iterations (36.86 M steps) with curriculum learning

## Demos

| Flat Terrain | Rough Terrain |
|-------------|--------------|
| [📹 Watch Video](videos/flat/final%20policy%20800%20learning%20iterations.mp4) | [📹 Watch Video](videos/rough/final%20policy%201500%20learning%20iterations.mp4) |
| 800 PPO iterations | 1500 PPO iterations |

## Repository Structure

- **modified codes/** – Our custom environment and PPO configuration files (`flat_env_cfg.py`, `rough_env_cfg.py`, `rsl_rl_ppo_cfg.py`, `velocity_env_cfg.py`).
- **training curves and logs/** – All training curves, exported CSV data, TensorBoard logs, and final plots.
- **videos/** – Recordings of the final policies on flat and rough terrain.
- **ECE489_Go1_Final_Presentation.pptx** – Project slide deck.
- **final_report.tex** – LaTeX source of the full report.

## How to Use

Make sure you have NVIDIA's **[Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/index.html)** and **[Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/index.html#)** installed with the RSL‑RL extension. Place the configuration files from `modified codes/` into the appropriate Isaac Lab task directories (see inline comments).

## Training and Visualization

### Rough terrain (training, 1500 iterations)

```cmd
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl/train.py --task Isaac-Velocity-Rough-Unitree-Go1-v0 --num_envs 1024 --device cuda:0 --max_iterations 1500 --seed 42 --video --video_length 200 --video_interval 100 --headless
```

### Flat terrain (trainig, 800 iterations)

```cmd
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl/train.py --task Isaac-Velocity-Flat-Unitree-Go1-v0 --num_envs 1024 --device cuda:0 --max_iterations 800 --seed 42 --video --headless
```

### Rough terrain (visualization)

```cmd
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl/play.py --task Isaac-Velocity-Rough-Unitree-Go1-v0 --num_envs 512 --device cuda:0 --checkpoint logs\rsl_rl\unitree_go1_rough\2026-05-25_17-21-50\model_1499.pt --video --video_length 1000 --enable_cameras --headless
```

### Flat terrain

```cmd
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl/play.py --task Isaac-Velocity-Flat-Unitree-Go1-v0 --num_envs 512 --device cuda:0 --checkpoint logs\rsl_rl\unitree_go1_flat\2026-05-25_22-04-58\model_799.pt --video --video_length 1000 --enable_cameras --headless
```

### Viewing training curves

```cmd
tensorboard --logdir=C:\IsaacLab\logs\rsl_rl
```

## Citation

If you use this work, please cite:

```bibtex
@misc{Soghomonyan_Pan_2026_quadruped,
  author       = {Hayk Soghomonyan and Siqi Pan},
  title        = {Learning-Based Quadruped Locomotion via NVIDIA Isaac Lab and Proximal Policy Optimization},
  institution  = {Zhejiang University, ECE 489 / ME 446},
  year         = {2026},
  howpublished = {\url{https://github.com/H-Sogh/ZJU_ECE489_final_project-Learning-Based_RL_Locomotion_Policy.git}},
}
```
