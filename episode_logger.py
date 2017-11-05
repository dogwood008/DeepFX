
# coding: utf-8

# In[ ]:


import rl.callbacks
import warnings
import timeit
import json
from tempfile import mkdtemp

import numpy as np


# In[ ]:


# https://github.com/matthiasplappert/keras-rl/blob/3cfe1f16b3d4911f3c8270880a8e2ac75180a136/rl/callbacks.py#L104
class EpisodeLogger(rl.callbacks.TrainEpisodeLogger):
    def __init__(self, logger):
        self._logger = logger
        super().__init__()

    def on_train_begin(self, logs):
        self.train_start = timeit.default_timer()
        self.metrics_names = self.model.metrics_names
        self._logger.critical('Training for {} steps ...'.format(self.params['nb_steps'])) # ここのみ変更

    def on_train_end(self, logs):
        duration = timeit.default_timer() - self.train_start
        self._logger.critical('done, took {:.3f} seconds'.format(duration)) # ここのみ変更
        
    def on_episode_end(self, episode, logs):
        duration = timeit.default_timer() - self.episode_start[episode]
        episode_steps = len(self.observations[episode])

        # Format all metrics.
        metrics = np.array(self.metrics[episode])
        metrics_template = ''
        metrics_variables = []
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            for idx, name in enumerate(self.metrics_names):
                if idx > 0:
                    metrics_template += ', '
                try:
                    value = np.nanmean(metrics[:, idx])
                    metrics_template += '{}: {:f}'
                except Warning:
                    value = '--'
                    metrics_template += '{}: {}'
                metrics_variables += [name, value]          
        metrics_text = metrics_template.format(*metrics_variables)

        nb_step_digits = str(int(np.ceil(np.log10(self.params['nb_steps']))) + 1)
        template = '{step: ' + nb_step_digits + 'd}/{nb_steps}: episode: {episode}, duration: {duration:.3f}s, episode steps: {episode_steps}, steps per second: {sps:.0f}, episode reward: {episode_reward:.3f}, mean reward: {reward_mean:.3f} [{reward_min:.3f}, {reward_max:.3f}], mean action: {action_mean:.3f} [{action_min:.3f}, {action_max:.3f}], mean observation: {obs_mean:.3f} [{obs_min:.3f}, {obs_max:.3f}], {metrics}'
        variables = {
            'step': self.step,
            'nb_steps': self.params['nb_steps'],
            'episode': episode + 1,
            'duration': duration,
            'episode_steps': episode_steps,
            'sps': float(episode_steps) / duration,
            'episode_reward': np.sum(self.rewards[episode]),
            'reward_mean': np.mean(self.rewards[episode]),
            'reward_min': np.min(self.rewards[episode]),
            'reward_max': np.max(self.rewards[episode]),
            'action_mean': np.mean(self.actions[episode]),
            'action_min': np.min(self.actions[episode]),
            'action_max': np.max(self.actions[episode]),
            'obs_mean': np.mean(self.observations[episode]),
            'obs_min': np.min(self.observations[episode]),
            'obs_max': np.max(self.observations[episode]),
            'metrics': metrics_text,
        }
        self._logger.error(template.format(**variables)) # ここのみ変更

        # Free up resources.
        del self.episode_start[episode]
        del self.observations[episode]
        del self.rewards[episode]
        del self.actions[episode]
        del self.metrics[episode]

    # def __init__(self):
    #     self.observations = {}
    #     self.rewards = {}
    #     self.actions = {}
# 
    # def on_episode_begin(self, episode, logs):
    #     self.observations[episode] = []
    #     self.rewards[episode] = []
    #     self.actions[episode] = []
# 
    # def on_step_end(self, step, logs):
    #     episode = logs['episode']
    #     self.observations[episode].append(logs['observation'])
    #     self.rewards[episode].append(logs['reward'])
    #     self.actions[episode].append(logs['action'])

