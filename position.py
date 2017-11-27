
# coding: utf-8

# In[ ]:


from action import Action


# In[ ]:


''' ポジション '''
class Position:
    def __init__(self, buy_or_sell, price, amount):
        self.price = price
        self.amount = amount
        self.buy_or_sell = buy_or_sell
    
    ''' 評価額を計算する '''
    def estimated_value(self, now_price):
        return now_price * self.amount

