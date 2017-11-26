
# coding: utf-8

# In[ ]:


from rl.callbacks import TrainEpisodeLogger
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')
import warnings
import timeit
import json

import numpy as np


# In[ ]:


class TestOutputLogger(TrainEpisodeLogger):
    def __init__(self, hist_data):
        self.logs = []
        self.hist_data = hist_data
        super(TestOutputLogger, self).__init__()

    def on_train_begin(self, logs):
        pass

    def on_train_end(self, logs):
        pass

    def on_episode_begin(self, episode, logs):
        pass

    def on_step_end(self, step, logs):
        self.logs.append(logs)
        #print(logs)

    def on_episode_end(self, episode, logs):
        x = self.hist_data.data().index
        y_reward = [step['reward'] for step in self.logs]
        y_price =  self.hist_data.data().loc[:, ['Close']].values
        y_reward.insert(0, np.nan)
        
        fig, ax1 = plt.subplots()
        ax1.plot(y_reward)
        ax2 = ax1.twinx()
        ax2.plot(y_price)
        plt.show()

