
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from logging import getLogger, StreamHandler, DEBUG, INFO
import time
import os
import warnings
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

from debug_tools import DebugTools
from hist_data import HistData
from episode_logger import EpisodeLogger
from model_saver import ModelSaver
from my_tensor_board import MyTensorBoard
from fx_trade import FXTrade


# In[ ]:


class DeepFX:
    def __init__(self, env, steps=50000,
              log_directory='./logs', model_directory='./models',
              model_filename='Keras-RL_DQN_FX_model_meanq{mean_q:e}_episode{episode:05d}',
              prepared_model_filename=None,
              weights_filename='Keras-RL_DQN_FX_weights.h5',):

        self._log_directory = log_directory
        self._model_directory = model_directory
        self._model_filename = model_filename
        self._prepared_model_filename = prepared_model_filename
        self._weights_filename = weights_filename
        self._load_model_path = self._relative_path(model_directory, prepared_model_filename) 
        self._save_model_path = self._relative_path(model_directory, model_filename)
        self._env = env
        self.steps = steps
        

    def setup(self):
        self._agent, self._model, self._memory, self._policy = self._initialize_agent()
        self._agent.compile('adam')
        print(self._model.summary())

    def train(self, is_for_time_measurement=False, wipe_instance_variables_after=True):
        self.setup()
        self._callbacks = self._get_callbacks()
        self._fit(self._agent, is_for_time_measurement, self._env, self._callbacks)
        if wipe_instance_variables_after:
            self._wipe_instance_variables()

    def test(self, episodes, callbacks=[], wipe_instance_variables_after=True):
        self.setup()
        self._agent.test(self._env, nb_episodes=episodes, visualize=False, callbacks=callbacks)

        get_ipython().magic('matplotlib inline')
        import matplotlib.pyplot as plt

        for obs in callbacks[0].rewards.values():
            plt.plot([o for o in obs])
        plt.xlabel("step")
        plt.ylabel("reward")
        if wipe_instance_variables_after:
            self._wipe_instance_variables()
        
    def _wipe_instance_variables(self):
         self._callbacks, self._agent, self._model,                 self._memory, self._policy, self.env = [None] * 6
        
    def _relative_path(self, directory, filename):
        if directory is None or filename is None:
            return None
        return os.path.join(directory, filename)

    def _get_model(self, load_model_path, observation_space_shape, nb_actions):
        if load_model_path is None:
            # DQNのネットワーク定義
            model = Sequential()
            model.add(Flatten(input_shape=(1,) + observation_space_shape))
            #model.add(Flatten(input_shape=observation_space_shape))
        #    model.add(Dense(4))
        #    model.add(Activation('relu'))
        #    model.add(Dense(4))
        #    model.add(Activation('relu'))
            model.add(Dense(nb_actions))
            model.add(Activation('relu'))
        else:
            model = keras.models.load_model(load_model_path)
        return model

    def _initialize_agent(self):
        nb_actions = self._env.action_space.n
        observation_space_shape = self._env.observation_space.shape
        model = self._get_model(self._load_model_path, observation_space_shape, nb_actions)
        
        # experience replay用のmemory
        memory = SequentialMemory(limit=500000, window_length=1)
        # 行動方策はオーソドックスなepsilon-greedy。ほかに、各行動のQ値によって確率を決定するBoltzmannQPolicyが利用可能
        policy = EpsGreedyQPolicy(eps=0.1) 
        dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
                       policy=policy)
                       #target_model_update=1e-2, policy=policy)
        #dqn.compile(Adam(lr=1e-3))
        return (dqn, model, memory, policy)
        
    def _get_callbacks(self):
        tensor_board_callback = MyTensorBoard(log_dir=self._log_directory, histogram_freq=1, embeddings_layer_names=True, write_graph=True)
        model_saver_callback = ModelSaver(self._save_model_path, monitor='mean_q', mode='max')
        callbacks = [tensor_board_callback, model_saver_callback]
        return callbacks

    def _fit(self, agent, is_for_time_measurement, env, callbacks=[]):
        if is_for_time_measurement:
            start = time.time()
            print(DebugTools.now_str())
            history = agent.fit(env, nb_steps=self.steps, visualize=False, verbose=2, nb_max_episode_steps=None,                              callbacks=callbacks)
            elapsed_time = time.time() - start
            print(("elapsed_time:{0}".format(elapsed_time)) + "[sec]")
            print(DebugTools.now_str())
        else:
            history = agent.fit(env, nb_steps=50000, visualize=True, verbose=2, nb_max_episode_steps=None)
        #学習の様子を描画したいときは、Envに_render()を実装して、visualize=True にします,
        
    def _render(self, mode='human', close=False):
        import pdb; pdb.set_trace()

