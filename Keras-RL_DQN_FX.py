
# coding: utf-8

# [[Python] Keras-RLで簡単に強化学習(DQN)を試す](http://qiita.com/inoory/items/e63ade6f21766c7c2393)を参考に、エージェントを作成する。FXの自動取引を行い、利益を出すのが目標。

# In[1]:

import gym
import gym.spaces
import numpy as np
import pandas as pd
import datetime
import enum
from logging import getLogger, StreamHandler, DEBUG, INFO


# In[2]:

#class Loglevel(enum.Enum):
#    DEBUG = 0; INFO = 1; NOTICE = 2; WARN = 3; ERROR = 3


# In[3]:

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(INFO)
logger.setLevel(INFO)
logger.addHandler(handler)

#Loglevel = enum.Enum('Loglevel', 'DEBUG INFO NOTICE WARN ERROR')
class Action(enum.Enum):
    SELL = -1; STAY = 0; BUY = +1


# In[4]:

class HistData:
    def __init__(self, date_range=None):
        self.csv_path = 'USDJPY.hst_.csv'
        self.csv_data = pd.read_csv(self.csv_path, index_col=0, parse_dates=True, header=0)
        self.date_range = date_range
    def data(self):
        if self.date_range is None:
            return self.csv_data
        else:
            return self.csv_data[self.date_range]
    def max_value(self):
        return self.data()[['High']].max()['High']
    def min_value(self):
        return self.data()[['Low']].min()['Low']
    def dates(self):
        return self.data().index.values
    def get_values_at(self, datetime):
        return self.data().loc[datetime]


# In[5]:

h = HistData('2010/09')


# In[6]:

# https://chrisalbon.com/python/pandas_time_series_basics.html
# h.data()['2010/09']


# In[7]:

#np.datetime64('2010-09-30T23:59:00.000000000+0000').total_seconds()
import numpy as np
np.version.full_version


# In[8]:

''' ポジション '''
class Position:
    def __init__(self, buy_or_sell, price, amount):
        self.price = price
        self.amount = amount
        self.buy_or_sell = buy_or_sell
    
    ''' 総利益を計算する '''
    def calc_profit_by(self, now_price):
        return self._calc_unit_profit_by(now_price) * self.amount

    ''' 単位あたりの利益を計算する '''
    def _calc_unit_profit_by(self, now_price):
        if self.buy_or_sell == 'buy' or self.buy_or_sell == Action.BUY.value:
            return now_price - self.price
        else:
            return self.price - now_price


# In[21]:

class FXTrade(gym.core.Env):
    AMOUNT_UNIT = 50000
    def __init__(self, initial_cash, spread, hist_data, seed_value=100000, logger=None):
        self.hist_data = hist_data
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.spread = spread
        self._positions = []
        self._max_date = self._datetime2float(hist_data.dates().max())
        self._min_date = self._datetime2float(hist_data.dates().min())
        self._seed = seed_value
        #logger.debugger = logger
        #logger.debug(self._max_date, Loglevel.DEBUG)
        #logger.debug(self._min_date, Loglevel.DEBUG)

        high = np.array([self._max_date, hist_data.max_value()])
        low = np.array([self._min_date, hist_data.min_value()])
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low = low, high = high) # DateFrame, Close prise

    '''def _logger(self, text, loglevel=Loglevel.DEBUG):
        if logger.debugger is not None:
            logger.debugger.log(text, level=loglevel)'''
        
    #@now_datetime.setter
    def get_now_datetime_as(self, datetime_or_float):
        if datetime_or_float == 'float':
            return self._now_datetime
        else:
            dt = self._float2datetime(self._now_datetime)
            return dt
    
    #@property
    def _set_now_datetime(self, value):
        if isinstance(value, float):
            assert self._min_date <= value, value
            assert value <= self._max_date, value
            self._now_datetime = value
            return value
        else:
            assert self._min_date <= self._datetime2float(value), '%f <= %f, %s' % (self._min_date, self._datetime2float(value), value)
            assert self._datetime2float(value) <= self._max_date, '%f <= %f, %s' % (self._datetime2float(value), self._max_date, value)
            float_val = self._datetime2float(value)
            self._now_datetime = float_val
            return float_val
            
    def setseed(self, seed_value):
        self._seed = seed_value
        np.random.seed(self._seed)
        print('Set seed value: %d' % self._seed)
        return seed_value
        
    def _seed(self):
        return self._seed
    
    def _datetime2float(self, datetime64_value):
        try:
            #import pdb; pdb.set_trace()
            float_val = float(str(datetime64_value.astype('uint64'))[:10])
            return float_val
        except:
            logger.error('_datetime2float except')
            import pdb; pdb.set_trace()
    
    def _float2datetime(self, float_timestamp):
        try:
            #import pdb; pdb.set_trace()
            datetime_val = np.datetime64(datetime.datetime.utcfromtimestamp(float_timestamp))
            return datetime_val
        except:
            logger.error('_float2datetime except')
            import pdb; pdb.set_trace()
    
    ''' 総含み益を計算する '''
    def _calc_total_unrealized_gain_by(self, now_close_value):
        if not self._positions: # positions is empty
            return 0
        total_profit = 0
        for position in self._positions:
            total_profit += position.calc_profit_by(now_close_value)
        return total_profit
    
    ''' 全ポジションを決済する '''
    def _liquidate_all_positions_by(self, now_price):
        total_profit = 0
        buy_or_sell = self._positions[0].buy_or_sell
        
        for position in self._positions:
            total_profit += position.calc_profit_by(now_price)
        self._positions = []
        self.cash += total_profit
        return total_profit
        
    ''' 注文を出す '''
    def _order(self, buy_or_sell, now_price, amount):
        position = Position(buy_or_sell=buy_or_sell, price=now_price, amount=amount)
        self._positions.append(position)
        return position
    
    def _get_price_of(self, buy_or_sell, now_buy_price, now_sell_price):
        if buy_or_sell == Action.BUY.value or buy_or_sell == Action.STAY.value:
            return now_buy_price
        elif buy_or_sell == Action.SELL.value:
            return now_sell_price
        else:
            return None
    
    ''' 各stepごとに呼ばれる
        actionを受け取り、次のstateとreward、episodeが終了したかどうかを返すように実装 '''
    def _step(self, action):
        logger.debug('_step STARTED')
        
        # actionを受け取り、次のstateを決定
        buy_or_sell_or_stay = action - 1
        assert buy_or_sell_or_stay == -1 or             buy_or_sell_or_stay == 0 or             buy_or_sell_or_stay == 1, 'buy_or_sell_or_stay: %d' % buy_or_sell_or_stay
        
        dminute = 1 # minuteの増加量
        # 今注目している日時を更新
        logger.debug('今注目している日時を更新')
        #import pdb; pdb.set_trace()
        now_datetime = self.get_now_datetime_as('datetime') + np.timedelta64(dminute, 'm')
        logger.debug('before')
        logger.debug(now_datetime)
        logger.debug(self._now_datetime)
        #import pdb; pdb.set_trace()
        self._set_now_datetime(now_datetime)
        logger.debug('after')
        logger.debug(now_datetime)
        logger.debug(self._now_datetime)
        #import pdb; pdb.set_trace()
        # その時点における値群
        now_values = self.hist_data.get_values_at(now_datetime)
        now_buy_price = now_values['Close']
        now_sell_price = now_buy_price - self.spread
        print('%s %f' % (now_datetime, now_buy_price))
        
        now_price = self._get_price_of(buy_or_sell_or_stay, now_buy_price, now_sell_price)
        if self._positions: # position is not empty
            if buy_or_sell_or_stay == Action.SELL.value:
                if self._positions[0].buy_or_sell == Action.BUY.value:
                    self._liquidate_all_positions_by(now_price)
                else:
                    self._order(buy_or_sell_or_stay, self.AMOUNT_UNIT, now_price)
            elif buy_or_sell_or_stay == Action.BUY.value:
                if self._positions[0].buy_or_sell == Action.SELL.value:
                    self._liquidate_all_positions_by(now_price)
                else:
                    self._order(buy_or_sell_or_stay, self.AMOUNT_UNIT, now_price)
        else: #position is empty
            if buy_or_sell_or_stay != Action.STAY:
                self._order(buy_or_sell_or_stay, self.AMOUNT_UNIT, now_price)
            
        
        # 現在の総含み益を再計算
        positions_buy_or_sell = None
        if self._positions:
            logger.debug('現在の総含み益を再計算')
            positions_buy_or_sell = self._positions[0].buy_or_sell
            logger.debug('buy_or_sell: %d' % positions_buy_or_sell)
        else:
            positions_buy_or_sell = Action.BUY.value
        logger.debug('positions_buy_or_sell: %d', positions_buy_or_sell)
        now_price_for_positions = self._get_price_of(positions_buy_or_sell, now_buy_price, now_sell_price)
        self._total_unrealized_gain = self._calc_total_unrealized_gain_by(now_price_for_positions)

        # 日付が学習データの最後と一致すれば終了
        done = now_datetime == self.hist_data.dates()[-1]
        if done:
            print('now_datetime: %s' % now_datetime)
            print('self.hist_data.dates()[-1]: %s' % self.hist_data.dates()[-1])
        else:
            print('now_datetime: %s' % now_datetime)
            print('self.hist_data.dates()[-1]: %s' % self.hist_data.dates()[-1])
            assert False

        # 報酬は現金と総含み益
        reward = self._total_unrealized_gain + self.cash
        
        # 次のstate、reward、終了したかどうか、追加情報の順に返す
        # 追加情報は特にないので空dict
        logger.debug('_step ENDED')
        return np.array([self._now_datetime, self._now_buy_price]), reward, done, {}
        
    ''' 各episodeの開始時に呼ばれ、初期stateを返すように実装 '''
    def _reset(self):
        print('_reset START')
        print(self.hist_data.dates()[0])

        self._set_now_datetime(self.hist_data.dates()[0])
        print(self._now_datetime)
        self._now_buy_price = self.hist_data.data()['Close'][0]
        self._positions = []
        print(self._seed)
        print('_reset END')
        return np.array([self._now_datetime, self._now_buy_price])
    


# In[20]:

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

env = FXTrade(1000000, 0.08, h)
nb_actions = env.action_space.n

# DQNのネットワーク定義
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# experience replay用のmemory
memory = SequentialMemory(limit=50000, window_length=1)
# 行動方策はオーソドックスなepsilon-greedy。ほかに、各行動のQ値によって確率を決定するBoltzmannQPolicyが利用可能
policy = EpsGreedyQPolicy(eps=0.1) 
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

history = dqn.fit(env, nb_steps=50000, visualize=False, verbose=2, nb_max_episode_steps=300)
#学習の様子を描画したいときは、Envに_render()を実装して、visualize=True にします,


# ## 現在の問題点その1
# `h = HistData('2010/09')` として、2010年の9月分を学習用に与えているが、ログを見る限り2010年9月1日3:20:00迄しか学習していない？

# In[11]:

model.save('Keras-RL_DQN_FX_model.h5')


# In[12]:

history


# In[13]:

model.save('Keras-RL_DQN_FX_weights.h5')


# In[14]:

import rl.callbacks
class EpisodeLogger(rl.callbacks.Callback):
    def __init__(self):
        self.observations = {}
        self.rewards = {}
        self.actions = {}

    def on_episode_begin(self, episode, logs):
        self.observations[episode] = []
        self.rewards[episode] = []
        self.actions[episode] = []

    def on_step_end(self, step, logs):
        episode = logs['episode']
        self.observations[episode].append(logs['observation'])
        self.rewards[episode].append(logs['reward'])
        self.actions[episode].append(logs['action'])

cb_ep = EpisodeLogger()
dqn.test(env, nb_episodes=10, visualize=False, callbacks=[cb_ep])


get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

for obs in cb_ep.observations.values():
    plt.plot([o[0] for o in obs])
plt.xlabel("step")
plt.ylabel("pos")


# ## 現在の問題点その2
# 2010年9月3日のデータは23:00:00迄しかなく、23:01:00を読み出そうとした時にエラーが発生している。適切にスキップする処理が必要か。

# In[ ]:

h.data()['2010-09-03']


# In[ ]:



