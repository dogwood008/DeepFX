
# coding: utf-8

# In[ ]:


import gym
import gym.spaces
import numpy as np
import datetime as dt
import time
from action import Action
from position import Position


# In[ ]:


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
        self._logger = logger
        np.random.seed(seed_value)

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
        print('Set seed value: %d' % self._seed)
        return seed_value
        
    def _seed(self):
        return self._seed
    
    def _datetime2float(self, datetime64_value):
        try:
            float_val = float(str(datetime64_value.astype('uint64'))[:10])
            return float_val
        except:
            self._logger.error('_datetime2float except')
            import pdb; pdb.set_trace()
    
    def _float2datetime(self, float_timestamp):
        try:
            datetime_val = np.datetime64(dt.datetime.utcfromtimestamp(float_timestamp))
            return datetime_val
        except:
            self._logger.error('_float2datetime except')
            import pdb; pdb.set_trace()
    
    ''' 総含み益を計算する '''
    def _calc_total_unrealized_gain_by(self, now_buy_price, now_sell_price):
        positions_buy_or_sell = None
        if self._positions:
            self._logger.debug('現在の総含み益を再計算')
            positions_buy_or_sell = self._positions[0].buy_or_sell
            self._logger.debug('buy_or_sell: %d' % positions_buy_or_sell)
        else:
            positions_buy_or_sell = Action.BUY.value
        self._logger.debug('positions_buy_or_sell: %d', positions_buy_or_sell)
        now_price_for_positions = self._get_price_of(positions_buy_or_sell, now_buy_price, now_sell_price)
        
        if not self._positions: # positions is empty
            return 0
        total_profit = 0
        for position in self._positions:
            total_profit += position.calc_profit_by(now_price_for_positions)
        return total_profit
    
    ''' 全ポジションを決済する '''
    def _close_all_positions_by(self, now_price):
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
    
    ''' 参照すべき価格を返す。取引しようとしているのが売りか買いかで判断する。 '''
    def _get_price_of(self, buy_or_sell, now_buy_price, now_sell_price):
        if buy_or_sell == Action.BUY.value or buy_or_sell == Action.STAY.value:
            return now_buy_price
        elif buy_or_sell == Action.SELL.value:
            return now_sell_price
        else:
            return None

    ''' 今注目している日時を1分進める '''
    def _increment_datetime(self):
        self._logger.debug('今注目している日時を更新 (=インデックスのインクリメント)')
        before_datetime = self.hist_data.data().iloc[[self._now_index], :].index[0]
        self._logger.debug('  before: %06d [%s]' % (self._now_index, before_datetime))
        self._now_index += 1
        after_datetime = self.hist_data.data().iloc[[self._now_index], :].index[0]
        self._logger.debug('   after: %06d [%s]' % (self._now_index, after_datetime))
        
    ''' For Debug: 毎日00:00に買値を表示する。学習の進捗を確認するため。 '''
    def _print_if_a_day_begins(self, now_datetime, now_buy_price):
        if now_datetime.hour == 0 and now_datetime.minute == 0:
            self._logger.info('%s %f' % (now_datetime, now_buy_price))
    
    ''' ポジションの手仕舞い、または追加オーダーをする '''
    def _close_or_more_order(self, buy_or_sell_or_stay, now_price):
        if not self._positions: # position is empty
            if buy_or_sell_or_stay != Action.STAY.value:
                self._order(buy_or_sell_or_stay, self.AMOUNT_UNIT, now_price)
        else: # I have positions
            # 売り: -1 / 買い: +1のため、(-1)の乗算で逆のアクションになる
            reverse_action = buy_or_sell_or_stay * (-1)
            if self._positions[0].buy_or_sell == reverse_action:
                # ポジションと逆のアクションを指定されれば、手仕舞い
                self._close_all_positions_by(now_price)
            else:
                # 追加オーダー
                self._order(buy_or_sell_or_stay, self.AMOUNT_UNIT, now_price)
        
    ''' 各stepごとに呼ばれる
        actionを受け取り、次のstateとreward、episodeが終了したかどうかを返すように実装 '''
    def _step(self, action):
        self._logger.debug('_step %06d STARTED' % self._now_index)
        
        # actionを受け取り、次のstateを決定
        buy_or_sell_or_stay = action - 1
        assert buy_or_sell_or_stay == -1 or             buy_or_sell_or_stay == 0 or             buy_or_sell_or_stay == 1, 'buy_or_sell_or_stay: %d' % buy_or_sell_or_stay
        
        # 今注目している日時を更新
        self._increment_datetime()
        
        # その時点における値群
        now_buy_price = self.hist_data.data().ix[[self._now_index], ['Close']].Close.iloc[0]
        now_sell_price = now_buy_price - self.spread
        
        # For Debug: 毎日00:00に買値を表示する。学習の進捗を確認するため。
        now_datetime = self.hist_data.data().iloc[[self._now_index], :].index[0]
        self._print_if_a_day_begins(now_datetime, now_buy_price)
        
        # actionによって、使用する価格を変える（売価/買価）
        now_price = self._get_price_of(buy_or_sell_or_stay,
                                       now_buy_price = now_buy_price,
                                       now_sell_price = now_sell_price)
        
        # ポジションの手仕舞い、または追加オーダーをする
        self._close_or_more_order(buy_or_sell_or_stay, now_price)
        
        # 現在の総含み益の合計値を再計算
        total_unrealized_gain = self._calc_total_unrealized_gain_by(
            now_buy_price, now_sell_price)

        # 日付が学習データの最後と一致すれば終了
        done = self._now_index >= len(self.hist_data.data()) - 1
        if done:
            print('now_datetime: %s' % now_datetime)
            print('len(self.hist_data.data()) - 1: %d' % (len(self.hist_data.data()) - 1))

        # 報酬は現金と総含み益
        reward = total_unrealized_gain + self.cash
        
        # 次のstate、reward、終了したかどうか、追加情報の順に返す
        # 追加情報は特にないので空dict
        self._logger.debug('_step ENDED')
        return np.array([self._now_index, now_buy_price]), reward, done, {}
        
    ''' 各episodeの開始時に呼ばれ、初期stateを返すように実装 '''
    def _reset(self):
        print('_reset START')
        print('self._seed: %i' % self._seed)
        initial_index = 0
        
        print('Start datetime: %s' % self.hist_data.dates()[initial_index])
        now_buy_price = self.hist_data.data().ix[[initial_index], ['Close']].Close.iloc[0]
        self._now_index = initial_index
        self._positions = []
        print('_reset END')
        next_state = [self._now_index, now_buy_price]
        return np.array(next_state)
    

