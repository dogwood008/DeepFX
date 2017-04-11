
# coding: utf-8

# [[Python] Keras-RLで簡単に強化学習(DQN)を試す](http://qiita.com/inoory/items/e63ade6f21766c7c2393)を参考に、エージェントを作成する。FXの自動取引を行い、利益を出すのが目標。

# In[42]:

import gym
import gym.spaces
import numpy as np
import pandas as pd
import datetime as dt
import enum
from logging import getLogger, StreamHandler, DEBUG, INFO
import time
import keras
import os
import warnings
import itertools


# In[43]:

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(INFO)
logger.setLevel(INFO)
logger.addHandler(handler)

class DebugTools:
    def now():
        return dt.datetime.now() + dt.timedelta(hours=9)
    def now_str():    
        return DebugTools.now().strftime('%y/%m/%d %H:%M:%S')

class Action(enum.Enum):
    SELL = -1; STAY = 0; BUY = +1


# In[44]:

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

    ''' 引数の日時がデータフレームに含まれるか '''
    def has_datetime(self, datetime64_value):
        try:
            h.data().loc[datetime64_value]
            return True
        except KeyError:
            return False

    def _get_nearist_index(self, before_or_after, datetime):
        if before_or_after == 'before':
            offset = -1
        else:
            offset = 0
        index = max(h.data().index.searchsorted(datetime) + offset, 0)
        return self.data().ix[self.data().index[index]]

    ''' 引数の日時を含まない直前に存在する値を取得する '''        
    def get_last_exist_datetime(self, datetime64_value):
        return self._get_nearist_index('before', datetime64_value)
        
    ''' 引数の日時を含む直後に存在する値を取得する '''
    def get_next_exist_datetime(self, datetime64_value):
        return self._get_nearist_index('after', datetime64_value)
    
    ''' fromとtoの日時の差が閾値内にあるか否か '''
    def is_datetime_diff_in_threshould(self, from_datetime, to_datetime, threshold_timedelta):
        last_datetime = h.get_last_exist_datetime(from_datetime)
        next_exist_datetime = h.get_next_exist_datetime(to_datetime)
        delta = next_exist_datetime.name - last_datetime.name
        return delta <= threshold_timedelta


# In[45]:

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


# In[46]:

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
                
        logger.debug('before')
        logger.debug(now_datetime)
        logger.debug(self._now_datetime)
        #import pdb; pdb.set_trace()
        self._set_now_datetime(now_datetime)
        logger.debug('after')
        logger.debug(now_datetime)
        logger.debug(self._now_datetime)
        # その時点における値群
        if h.has_datetime(now_datetime):
            modified_now_datetime = now_datetime
        else:
            modified_now_datetime = self.hist_data.get_last_exist_datetime(now_datetime).name
        now_values = self.hist_data.data().loc[modified_now_datetime]
        
        now_buy_price = now_values['Close']
        self._now_buy_price = now_buy_price
        now_sell_price = now_buy_price - self.spread
        
        logger.debug(modified_now_datetime)
        if pd.DatetimeIndex([modified_now_datetime]).hour[0] == 0 and            pd.DatetimeIndex([modified_now_datetime]).minute[0] == 0:
            # 毎日00:00に表示
            if now_datetime == modified_now_datetime:
                logger.info('%s %f' % (now_datetime, now_buy_price))
            else:
                 logger.info('%s (mod: %s) %f' % (now_datetime, modified_now_datetime, now_buy_price))
        
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
    


# In[66]:

import rl.callbacks
class ModelSaver(rl.callbacks.TrainEpisodeLogger):
    def __init__(self, filepath, monitor='loss', verbose=1, save_weights_only=True):
        self.min_monitor_value = None
        self.filepath = filepath
        self.monitor = monitor
        self.verbose = verbose
        #self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        super().__init__()

    def on_episode_end(self, episode, logs):
        print('========== Model Saver output ==============')
        loss_value = self._formatted_metrics(episode)[self.monitor]
        print('loss_value: %f' % loss_value)
        if self.min_monitor_value is None or loss_value < self.min_monitor_value:
            previous_value = self.min_monitor_value
            self.min_monitor_value = loss_value
            self._save_model(previous_monitor=previous_value, loss=loss_value, episode=episode)
        print('min monitor loss: %f' % self.min_monitor_value)
        print('========== /Model Saver output =============')
        super().on_episode_end(episode, logs)

    def _save_model(self, previous_monitor, loss, episode):
        filepath = self.filepath.format(loss=loss, episode=episode)
        if self.verbose > 0:
            print('Step %05d: model improved from %0.5f to %0.5f,'
                  ' saving model to %s'
                  % (self.step, previous_monitor or 0.0,
                     self.min_monitor_value or 0.0, filepath))
        if self.save_weights_only:
            self.model.save_weights(filepath, overwrite=True)
        else:
            raise NotImplementedError

    def _formatted_metrics(self, episode):
        # Format all metrics.
        metrics = np.array(self.metrics[episode])
        metrics_variables = []
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            for idx, name in enumerate(self.metrics_names):
                try:
                    value = np.nanmean(metrics[:, idx])
                except Warning:
                    value = '--'
                metrics_variables += [name, value]
        return dict(itertools.zip_longest(*[iter(metrics_variables)] * 2, fillvalue=""))
        


# In[56]:

h = HistData('2010/09/01')


# In[67]:

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

log_directory = './log'
model_directory = './model'
#model_filename = 'Keras-RL_DQN_FX_model{epoch:02d}-loss{loss:.2f}-acc{acc:.2f}-vloss{val_loss:.2f}-vacc{val_acc:.2f}.hdf5'
model_filename = 'Keras-RL_DQN_FX_model_episode{episode:05d}_loss{loss:e}.hdf5'
weights_filename = 'Keras-RL_DQN_FX_weights.h5'

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

tensor_board_callback = keras.callbacks.TensorBoard(log_dir=log_directory, histogram_freq=1)
check_point_callback = keras.callbacks.ModelCheckpoint(filepath = os.path.join(model_directory, model_filename),                                        monitor='metrics["loss"]', verbose=1, save_best_only=True, mode='auto')

model_saver_callback = ModelSaver(model_filename)

is_for_time_measurement = True
if is_for_time_measurement:
    start = time.time()
    print(DebugTools.now_str())
    #minutes = 2591940/60 # 2591940secs = '2010-09-30 23:59:00' - '2010-09-01 00:00:00'
    minutes = (60 * 24 - 1) * 2 # 2days
    history = dqn.fit(env, nb_steps=minutes, visualize=False, verbose=2, nb_max_episode_steps=None,                      callbacks=[model_saver_callback])
    elapsed_time = time.time() - start
    print(("elapsed_time:{0}".format(elapsed_time)) + "[sec]")
    print(DebugTools.now_str())
else:
    history = dqn.fit(env, nb_steps=50000, visualize=False, verbose=2, nb_max_episode_steps=None)
#学習の様子を描画したいときは、Envに_render()を実装して、visualize=True にします,


# In[50]:

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


# In[51]:

if False:
    cb_ep = EpisodeLogger()
    dqn.test(env, nb_episodes=10, visualize=False, callbacks=[cb_ep])


    get_ipython().magic('matplotlib inline')
    import matplotlib.pyplot as plt

    for obs in cb_ep.observations.values():
        plt.plot([o[0] for o in obs])
    plt.xlabel("step")
    plt.ylabel("pos")

