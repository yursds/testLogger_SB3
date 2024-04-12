import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from stable_baselines3          import PPO

#from trainCartPole import CartPoleEnv as Myenv
#model_str = "ppo_cartpole10K"

from trainPendulum import PendulumEnv as Myenv
model_str = "ppo_pendulum800k"

if __name__ == '__main__':
    
    model = PPO.load(model_str)
    env = Myenv(render_mode="human")

    obs, _ = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, _, info = env.step(action)
        if dones:
            break