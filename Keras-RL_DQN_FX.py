
# coding: utf-8

# [[Python] Keras-RLで簡単に強化学習(DQN)を試す](http://qiita.com/inoory/items/e63ade6f21766c7c2393)を参考に、エージェントを作成する。FXの自動取引を行い、利益を出すのが目標。

# In[9]:

import gym
import gym.spaces
import numpy as np
import pandas as pd
import datetime as dt
import enum
from logging import getLogger, StreamHandler, DEBUG, INFO


# In[10]:

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(INFO)
logger.setLevel(INFO)
logger.addHandler(handler)

class Action(enum.Enum):
    SELL = -1; STAY = 0; BUY = +1


# In[11]:

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

    def get_last_exist_datetime_recursively(self, datetime64_value):
        try:
            return self.data().loc[datetime64_value], datetime64_value
        except KeyError:
            if self.data().index[0] > datetime64_value:
                return self.data().iloc[0], self.data().index[0]
            previous_datetime = datetime64_value - np.timedelta64(1, 'm')
            return self.get_last_exist_datetime_recursively(previous_datetime)
        
    ''' 引数の日時を含む直後に存在する値を取得する '''
    def get_next_exist_datetime(self, datetime64_value):
        index_size = len(self.data())
        last_index = index_size - 1
        # KeyErrorは補足しない。範囲外にならないように学習の際に
        # `nb_max_episode_steps` を与える必要がある
        next_index = min(             self.data().index.get_loc(datetime64_value) + 1,              last_index)
        return self.data().iloc[next_index], self.data().index[next_index]
    
    ''' fromとtoの日時の差が閾値内にあるか否か '''
    def is_datetime_diff_in_threshould(self, from_datetime, to_datetime, threshold_timedelta):
        last_datetime, _ = h.get_last_exist_datetime_recursively(from_datetime)
        next_exist_datetime, _ = h.get_next_exist_datetime(to_datetime)
        delta = next_exist_datetime.name - last_datetime.name
        return delta <= threshold_timedelta


# ## 現在の問題点その3
# 
# 一つ心配事は、土日等休場日も学習すべきかどうかである。おそらく、４８時間全く値動きがないことを学習しても仕方ないので、これは飛ばして良いと思う。問題はその次の数分の欠測である。欠測の間は値動き無しとして学習するのが良いのか、純粋に経過時間（分）で学習するのが良いのかは、明確な答えを持っていない。一旦、閾値までの間値動きがなければ、次の値動きまでスキップするようにしようと思う。

# ### 現在の問題点その3に対する解
# 時間のある時にきちんとチェックする必要があるが、Udacityの[Machine Learning for Trading](https://www.udacity.com/course/machine-learning-for-trading--ud501)では、休場中を特別な扱いはしていなかったような記憶がある（間違っていたら修正する必要がある）。したがって、メソッド `is_datetime_diff_in_threshould()` を作ったが、今は使わないでおくことにする。

# In[12]:

h = HistData('2010/09')


# In[13]:

from_dt = np.datetime64('2010-09-03T23:01:00.000000')
to_dt = np.datetime64('2010-09-03T23:00:00.000000')
h.is_datetime_diff_in_threshould(from_dt, to_dt, dt.timedelta(days=2, hours=1, seconds=1))


# In[14]:

#np.datetime64('2010-09-30T23:59:00.000000000+0000').total_seconds()
import numpy as np
np.version.full_version


# In[15]:

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


# In[16]:

class FXTrade(gym.core.Env):
    AMOUNT_UNIT = 50000
    THRESHOULD_TIME_DELTA = dt.timedelta(days=1)
    
    def __init__(self, initial_cash, spread, hist_data, seed_value=100000, logger=None):
        self.hist_data = hist_data
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.spread = spread
        self._positions = []
        self._max_date = self._datetime2float(hist_data.dates().max())
        self._min_date = self._datetime2float(hist_data.dates().min())
        self._seed = seed_value

        high = np.array([self._max_date, hist_data.max_value()])
        low = np.array([self._min_date, hist_data.min_value()])
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low = low, high = high) # DateFrame, Close prise
        
    def get_now_datetime_as(self, datetime_or_float):
        if datetime_or_float == 'float':
            return self._now_datetime
        else:
            dt = self._float2datetime(self._now_datetime)
            return dt
    
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
            float_val = float(str(datetime64_value.astype('uint64'))[:10])
            return float_val
        except:
            logger.error('_datetime2float except')
            import pdb; pdb.set_trace()
    
    def _float2datetime(self, float_timestamp):
        try:
            datetime_val = np.datetime64(dt.datetime.utcfromtimestamp(float_timestamp))
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
        previous_datetime = self.get_now_datetime_as('datetime')
        now_datetime = previous_datetime + np.timedelta64(dminute, 'm')
        
        if now_datetime != modified_now_datetime:
            assert self.hist_data.is_datetime_diff_in_threshould(previous, now_datetime, THRESHOULD_TIME_DELTA)
        
        logger.debug('before')
        logger.debug(now_datetime)
        logger.debug(self._now_datetime)
        #import pdb; pdb.set_trace()
        self._set_now_datetime(now_datetime)
        logger.debug('after')
        logger.debug(now_datetime)
        logger.debug(self._now_datetime)
        # その時点における値群
        now_values, modified_now_datetime = self.hist_data.get_last_exist_datetime_recursively(now_datetime)
        
        now_buy_price = now_values['Close']
        self._now_buy_price = now_buy_price
        now_sell_price = now_buy_price - self.spread
        
        if pd.DatetimeIndex([modified_now_datetime]).minute[0] == 0:
            if now_datetime == modified_now_datetime:
                # 毎時00分に表示
                print('%s %f' % (now_datetime, now_buy_price))
            else:
                 print('%s (mod: %s) %f' % (now_datetime, modified_now_datetime, now_buy_price))
        
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
            assert False #FIXME


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
    


# In[14]:

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

history = dqn.fit(env, nb_steps=50000, visualize=False, verbose=2, nb_max_episode_steps=None)
#学習の様子を描画したいときは、Envに_render()を実装して、visualize=True にします,


# ## 現在の問題点その1
# `h = HistData('2010/09')` として、2010年の9月分を学習用に与えているが、ログを見る限り2010年9月1日3:20:00迄しか学習していない？

# ### 問題点その1に対する解
# 
# `nb_max_episode_steps = None` にする。これが各エピソードにおけるステップ数の上限になっていた。
# 
# `2010-09-01T05:00:00.000000+0000` で終了するのは、 60 min * 5 hours = 300 steps で上限に達していたからだった。
# 
# ソース: https://github.com/matthiasplappert/keras-rl/blob/master/rl/core.py#L280

# In[ ]:

len(h.data()['2010/09'])


# In[ ]:

model.save('Keras-RL_DQN_FX_model.h5')


# In[ ]:

history


# In[ ]:

model.save('Keras-RL_DQN_FX_weights.h5')


# In[ ]:

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

