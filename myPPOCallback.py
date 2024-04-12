from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO

import numpy as np

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, model:PPO, verbose=0):
        super().__init__(verbose)
        self.mean_episode_rewards = []
        self.mmodel = model
        
        
    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        value = np.random.random()
        self.logger.record("random_value", value)
        
        # Log reward
        if self.num_timesteps > 0:

            # Retrieve the reward from the rollout buffer
            reward = self.mmodel.rollout_buffer.rewards

            self.logger.record("reward", reward[0][0])
            if len(self.model.ep_info_buffer) > 0:
                mean_episode_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
            else:
                mean_episode_reward = 0.0
            self.mean_episode_rewards.append(mean_episode_reward)
            self.logger.record("mean_episode_reward", mean_episode_reward)
        return True