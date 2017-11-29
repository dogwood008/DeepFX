
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from logging import getLogger, StreamHandler, DEBUG, INFO
import time
import os
import warnings
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, InputLayer
from keras.layers.core import Permute
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.initializers import TruncatedNormal

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

from debug_tools import DebugTools
from hist_data import HistData
from episode_logger import EpisodeLogger
from model_saver import ModelSaver
from my_tensor_board import MyTensorBoard
from fx_trade import FXTrade
from debug_tools import DebugTools


# In[ ]:


class DeepFX:
    def __init__(self, env, steps=50000,
              log_directory='./logs', model_directory='./models',
              model_filename='%s_model_episode{episode:07d}_mae{mean_absolute_error:3.3e}_' % DebugTools.now_12(),
              prepared_model_filename=None,
              weights_filename='Keras-RL_DQN_FX_weights.h5',
              logger=None):

        self._log_directory = log_directory
        self._model_directory = model_directory
        self._model_filename = model_filename
        self._prepared_model_filename = prepared_model_filename
        self._weights_filename = weights_filename
        self._load_model_path = self._relative_path(model_directory, prepared_model_filename) 
        self._save_model_path = self._relative_path(model_directory, model_filename)
        self._env = env
        self.steps = steps
        self._logger = logger
        

    def setup(self):
        self._agent, self._model, self._memory, self._policy = self._initialize_agent()
        self._agent.compile(Adam(lr=.00025), metrics=['mae'])
        self._logger.info(self._model.summary())

    def train(self, is_for_time_measurement=False, wipe_instance_variables_after=True):
        self.setup()
        self._callbacks = self._get_callbacks()
        self._fit(self._agent, is_for_time_measurement, self._env, self._callbacks)
        if wipe_instance_variables_after:
            self._wipe_instance_variables()

    def test(self, callbacks=[], wipe_instance_variables_after=True):
        self.setup()
        self._agent.test(self._env, visualize=False, callbacks=callbacks)

        #%matplotlib inline
        #import matplotlib.pyplot as plt
#
        #for obs in callbacks[0].rewards.values():
        #    plt.plot([o for o in obs])
        #plt.xlabel("step")
        #plt.ylabel("reward")
        #if wipe_instance_variables_after:
        #    self._wipe_instance_variables()
        
    def _wipe_instance_variables(self):
         self._callbacks, self._agent, self._model,                 self._memory, self._policy, self.env = [None] * 6
        
    def _relative_path(self, directory, filename):
        if directory is None or filename is None:
            return None
        return os.path.join(directory, filename)

    def _get_model(self, load_model_path, observation_space_shape, nb_actions):
        import keras.backend as K
        if load_model_path is not None:
            model = keras.models.load_model(load_model_path)
            return model
        # DQNのネットワーク定義
        # ref: https://github.com/googledatalab/notebooks/blob/master/samples/TensorFlow/Machine%20Learning%20with%20Financial%20Data.ipynb
        # ref: https://elix-tech.github.io/ja/2016/06/29/dqn-ja.html
        # ref: https://github.com/matthiasplappert/keras-rl/blob/master/examples/dqn_atari.py
        # ref: https://github.com/yukiB/keras-dqn-test
        model = Sequential()
        model.add(Dense(128, activation='relu',
                        bias_initializer='ones',
                        input_shape=(48,) + observation_space_shape))
        model.add(Flatten())
        model.add(Dense(128, activation='relu',
                        bias_initializer='ones'))
        model.add(Dense(128, activation='relu',
                        bias_initializer='ones'))
        model.add(Dense(nb_actions, activation='linear'))
        return model

    def _initialize_agent(self):
        nb_actions = self._env.action_space.n
        observation_space_shape = self._env.observation_space.shape
        model = self._get_model(self._load_model_path, observation_space_shape, nb_actions)
        
        # experience replay用のmemory
        memory = SequentialMemory(limit=500000, window_length=48)
        # 行動方策はオーソドックスなepsilon-greedy。ほかに、各行動のQ値によって確率を決定するBoltzmannQPolicyが利用可能
        policy = EpsGreedyQPolicy(eps=0.1) 
        dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
                       policy=policy)
                       #target_model_update=1e-2, policy=policy)
        #dqn.compile(Adam(lr=1e-3))
        return (dqn, model, memory, policy)
        
    def _get_callbacks(self):
        tensor_board_callback = MyTensorBoard(log_dir=self._log_directory, histogram_freq=1, embeddings_layer_names=True, write_graph=True)
        model_saver_callback = ModelSaver(self._save_model_path, monitor='mean_absolute_error', mode='min', logger=self._logger, save_best_only=False)
        episode_logger_callback = EpisodeLogger(logger=self._logger)
        callbacks = [tensor_board_callback, model_saver_callback, episode_logger_callback]
        return callbacks

    def _fit(self, agent, is_for_time_measurement, env, callbacks=[]):
        if is_for_time_measurement:
            start = time.time()
            self._logger.info(DebugTools.now_str())
            history = agent.fit(env, nb_steps=self.steps, visualize=False, verbose=2, nb_max_episode_steps=None,                              callbacks=callbacks)
            elapsed_time = time.time() - start
            self._logger.warn(("elapsed_time:{0}".format(elapsed_time)) + "[sec]")
            self._logger.info(DebugTools.now_str())
        else:
            history = agent.fit(env, nb_steps=50000, visualize=True, verbose=2, nb_max_episode_steps=None)
        #学習の様子を描画したいときは、Envに_render()を実装して、visualize=True にします,
        
    def _render(self, mode='human', close=False):
        import pdb; pdb.set_trace()

