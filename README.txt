# DiffBot MuJoCo Reinforcement Learning

## Problem

Teach a two-wheel differential-drive robot to reach a target from different starting orientations.

## Solution

The robot is simulated in MuJoCo and wrapped as a Gymnasium environment. A PPO agent from Stable-Baselines3 learns to control forward and turning speed using the target position and robot motion. It receives rewards for moving closer and reaching the target.

## Run on Windows

Install Python 3.13, then run these commands in the project folder:

```powershell
python -m pip install mujoco gymnasium stable-baselines3
python train_diffbot.py
python watch_diffbot.py
```

Training saves a model and evaluates it on 20 targets. The viewer shows the robot driving to the green target.

## Current scope

This is an initial simulation with an empty floor and ideal simulated position information. Obstacle avoidance, functional LiDAR, and real-robot deployment are not yet implemented.
