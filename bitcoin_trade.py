
# coding: utf-8

# In[ ]:


import gym
import gym.spaces
import numpy as np
import datetime as dt
import time
from action import Action
from position import Position
from fx_trade import FXTrade


# In[ ]:


class BitcoinTrade(FXTrade):
    THRESHOULD_TIME_DELTA = dt.timedelta(days=1)
    
    def __init__(self, initial_cash, spread, hist_data, seed_value=100000,
                 logger=None, amount_unit=1):
        self.hist_data = hist_data
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.spread = spread
        self._positions = []
        self._seed = seed_value
        self._logger = logger
        self.amount_unit = amount_unit # 1BTC
        np.random.seed(seed_value)
        
        # x軸: N番目の足（時間経過）, y軸: 現在の1USDの価格（単位：円）
        high = np.array([self.hist_data.steps(), hist_data.data()['Close'].max()]) # [x軸最大値, y軸最大値]
        low = np.array([0, hist_data.data()['Close'].min()]) # [x軸最小値, y軸最小値]
        self.action_space = gym.spaces.Discrete(len(Action)) # Actionクラスで定義 買う、売る、なにもしないの3択
        self.observation_space = gym.spaces.Box(low = low, high = high) # [N番目の足, Close price]

    ''' ポジションの手仕舞い、または追加オーダーをする '''
    def _close_or_more_order(self, buy_or_sell_or_stay, now_price):
        if not self._positions: # position is empty
            #if buy_or_sell_or_stay != Action.STAY.value:
            # FXでは売り買いどちらかでもポジションを持てたが、
            # Bitcoinは買いからのみとする
            if buy_or_sell_or_stay == Action.BUY.value:
                self._order(buy_or_sell_or_stay,
                            now_price=now_price, amount=self.amount_unit)
        else: # I have positions
            # 売り: -1 / 買い: +1のため、(-1)の乗算で逆のアクションになる
            # reverse_action = buy_or_sell_or_stay * (-1)
            # if self._positions[0].buy_or_sell == reverse_action:
            # 売りサインなら手仕舞い
            if buy_or_sell_or_stay == Action.SELL.value:
                self._close_all_positions_by(now_price)
            elif buy_or_sell_or_stay == Action.BUY.value:
                # 買いサインなら買い増し追加オーダー
                self._order(buy_or_sell_or_stay,
                            now_price=now_price, amount=self.amount_unit)

