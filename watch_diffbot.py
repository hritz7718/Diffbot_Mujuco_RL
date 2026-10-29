from pathlib import Path
import time
from stable_baselines3 import PPO
from diffbot_env import DiffBotEnv

def main():
    pointer = Path(__file__).with_name('latest_model.txt')
    if not pointer.exists():
        raise SystemExit('No completed training run found. Run 2 - Train DiffBot.cmd first.')
    model = PPO.load(pointer.read_text().strip(), device='cpu')
    env = DiffBotEnv(render_mode='human')
    try:
        obs, _ = env.reset(seed=20000)
        while env.viewer.is_running():
            start = time.monotonic()
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print('Target reached!' if info['is_success'] else 'Attempt ended; trying another target.')
                obs, _ = env.reset()
            time.sleep(max(0, env.dt-(time.monotonic()-start)))
    finally:
        env.close()

if __name__ == '__main__':
    main()
