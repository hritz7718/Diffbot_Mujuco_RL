from pathlib import Path
from datetime import datetime
import argparse
import json
import math
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from diffbot_env import DiffBotEnv


def evaluate(env, predict, count=20, verbose=False):
    successes = 0
    for seed in range(10000, 10000+count):
        obs, _ = env.reset(seed=seed)
        start_distance = float(np.linalg.norm(obs[:2]))
        for _ in range(600):
            obs, _, term, trunc, info = env.step(predict(obs))
            if term or trunc:
                successes += int(info['is_success'])
                if verbose:
                    reason = 'reached' if info['is_success'] else ('timeout' if trunc else 'failed')
                    print(f'Target {seed}: {reason}; distance {start_distance:.2f} -> '
                          f'{info["distance"]:.2f} m; heading error '
                          f'{math.degrees(math.atan2(float(obs[1]), float(obs[0]))):.1f} deg; '
                          f'wheel speeds {obs[4]:.2f}, {obs[5]:.2f}', flush=True)
                break
    return successes/count


def baseline(obs):
    angle = math.atan2(float(obs[1]), float(obs[0]))
    distance = math.hypot(float(obs[0]), float(obs[1]))
    # Align before translating, and slow down near the target. The original
    # constant-speed approach could orbit a nearby target instead of reaching it.
    forward = min(0.8, 2*distance) if abs(angle) < 0.25 else 0.0
    turn = max(-1.0, min(1.0, 3*angle))
    return [forward, turn]


def motion_diagnostics(env):
    for name, action in [('forward', [0.7, 0]), ('left turn', [0, 0.7])]:
        env.reset(seed=42)
        start = env.data.qpos[:2].copy()
        r = env.data.body('base_link').xmat.reshape(3, 3)
        yaw0 = math.atan2(r[1, 0], r[0, 0])
        for _ in range(40):
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break
        r = env.data.body('base_link').xmat.reshape(3, 3)
        yaw = math.atan2(r[1, 0], r[0, 0])
        change = math.atan2(math.sin(yaw-yaw0), math.cos(yaw-yaw0))
        delta = env.data.qpos[:2]-start
        forward = math.cos(yaw0)*delta[0]+math.sin(yaw0)*delta[1]
        print(f'Motion check ({name}): forward {forward:.3f} m, '
              f'yaw change {math.degrees(change):.1f} deg, '
              f'wheel speeds {obs[4]:.2f}, {obs[5]:.2f}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=300000)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    env = DiffBotEnv()
    check_env(env, warn=True)
    # Verify seed reproducibility and both episode-ending paths.
    a, _ = env.reset(seed=42)
    b, _ = env.reset(seed=42)
    np.testing.assert_allclose(a, b)
    env.goal = env.data.qpos[:2].copy()
    _, _, terminated, _, info = env.step(np.zeros(2))
    assert terminated and info['is_success']
    env.reset(seed=42)
    env.steps = 599
    _, _, terminated, truncated, _ = env.step(np.zeros(2))
    assert truncated and not terminated
    print('Environment checks passed.', flush=True)
    motion_diagnostics(env)
    baseline_score = evaluate(env, baseline, count=10, verbose=True)
    print(f'Conventional controller: {baseline_score:.0%} success (10 targets).', flush=True)
    if baseline_score < 0.8:
        raise RuntimeError('Baseline failed too often. Inspect environment before training.')
    if args.check_only:
        env.close()
        return
    folder = Path(__file__).parent / 'runs' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    folder.mkdir(parents=True)
    # Each run is independent and preserves earlier models.
    import shutil
    for name in ['diffbot.xml', 'diffbot_env.py', 'train_diffbot.py']:
        shutil.copy2(Path(__file__).with_name(name), folder / name)
    import importlib.metadata
    versions = {p: importlib.metadata.version(p) for p in ['mujoco', 'gymnasium', 'stable-baselines3', 'torch', 'numpy']}
    (folder/'versions.json').write_text(json.dumps(versions, indent=2))
    training = Monitor(env, str(folder/'training'), info_keywords=('is_success',))
    model = PPO('MlpPolicy', training, device='cpu', verbose=1, seed=7,
                n_steps=2048, batch_size=64, gamma=0.995, learning_rate=3e-4,
                policy_kwargs={'net_arch': [64, 64]})
    print(f'Saving to {folder}', flush=True)
    try:
        model.learn(total_timesteps=args.steps, callback=CheckpointCallback(
            save_freq=25000, save_path=str(folder), name_prefix='checkpoint'))
    except KeyboardInterrupt:
        model.save(folder/'interrupted')
        print('Interrupted model saved. Training has not completed.')
        return
    finally:
        training.close()
    model.save(folder/'ppo_diffbot')
    evaluation = DiffBotEnv()
    score = evaluate(evaluation, lambda obs: model.predict(obs, deterministic=True)[0])
    evaluation.close()
    (folder/'evaluation.json').write_text(json.dumps({
        'success_rate': score, 'episodes': 20, 'seed_start': 10000,
        'baseline_success_rate_10_episodes': baseline_score,
        'timesteps': model.num_timesteps}, indent=2))
    Path(__file__).with_name('latest_model.txt').write_text(str(folder/'ppo_diffbot.zip'))
    print(f'PPO success rate: {score:.0%} on 20 test targets. Run 3 - Watch DiffBot.cmd.')


if __name__ == '__main__':
    main()
