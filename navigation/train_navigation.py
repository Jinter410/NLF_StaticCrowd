import time
import gymnasium as gym
import Gymnasium_envs
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

load_model = True
kwargs = {'n_rays': 40, 'max_steps': 150}
env_name = "Navigation-v0"
model_path = "./navigation/results/sac_Nav40Rays/models/rl_model_999984_steps.zip"
env_path = "./navigation/results/sac_Nav40Rays/models/rl_model_vecnormalize_999984_steps.pkl"
if load_model:
    kwargs['render_mode'] = 'human'
    env = VecNormalize.load(env_path, make_vec_env(env_name, n_envs=1, env_kwargs=kwargs))
    env.training = False
    env.norm_reward = False
    model = SAC.load(model_path, device='cpu')
    print("Model loaded")
    rewards = 0
    obs= env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        
        obs, reward, done, info = env.step(action)
        rewards += reward
        env.render()
        time.sleep(0.03)

        if done:
            print(rewards)
            rewards = 0
            obs = env.reset()
    env.close()

else:
    n_envs = 12
    name_prefix = "sac_Nav40Rays"
    log_dir = f"./navigation/results/{name_prefix}/logs"
    model_dir = f"./navigation/results/{name_prefix}/models"

    # Normalize
    checkpoint_callback = CheckpointCallback(save_freq=max(500_000 // n_envs, 1), save_path=model_dir, save_vecnormalize=True)
    vec_env = VecNormalize(make_vec_env(env_name, n_envs=12, env_kwargs=kwargs), norm_obs=True)
    model = SAC(
            "MlpPolicy", vec_env,
            learning_rate=3e-4,
            verbose=1,
            tensorboard_log=log_dir,
            device='cpu'
        )

    print("Training")
    model.learn(total_timesteps=2_000_000, callback=checkpoint_callback)
    print("Model saved")
