import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

class L2TradingEnv(gym.Env):
    """
    Tier-1 Institutional Custom Environment for High-Frequency Trading.
    Simulates Level 2 Order Book dynamics including Maker/Taker friction 
    and passive queue position heuristics.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, data_path, window_size=10, initial_balance=10000.0, taker_fee=0.0005, maker_rebate=0.0001, start_idx=None, end_idx=None):
        super(L2TradingEnv, self).__init__()
        
        # Load the reconstructed order book features
        self.df = pd.read_csv(data_path)
        
        # Apply Train/Test split
        if start_idx is not None or end_idx is not None:
            self.df = self.df.iloc[start_idx:end_idx].reset_index(drop=True)
        
        # Derive missing prices from spread
        self.df['bid_price_1'] = self.df['ask_price_1'] - self.df['spread']
        self.df['mid_price'] = self.df['ask_price_1'] - (self.df['spread'] / 2.0)
        
        # Keep only derived stationary features for the observation space
        feature_cols = ['obi', 'spread']
        self.features = self.df[feature_cols].values
        
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.taker_fee = taker_fee
        self.maker_rebate = maker_rebate
        
        # Action space: 
        # 0 = Hold
        # 1 = Market Buy (Aggressive, pays taker fee)
        # 2 = Market Sell (Aggressive, pays taker fee)
        # 3 = Limit Buy (Passive, posts at bid, earns maker rebate if filled)
        # 4 = Limit Sell (Passive, posts at ask, earns maker rebate if filled)
        self.action_space = spaces.Discrete(5)
        
        # Observation space
        self.n_features = self.features.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.window_size, self.n_features), dtype=np.float32
        )
        
        # State tracking
        self.current_step = 0
        self.max_steps = len(self.features) - 1
        self.balance = self.initial_balance
        self.position = 0 # 1 for Long, -1 for Short, 0 for Flat
        self.entry_price = 0.0
        
        # Pending Limit Orders
        self.pending_limit_buy = None
        self.pending_limit_sell = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0.0
        self.pending_limit_buy = None
        self.pending_limit_sell = None
        
        return self._get_observation(), {}
        
    def _get_observation(self):
        obs = self.features[self.current_step - self.window_size:self.current_step]
        return obs.astype(np.float32)
        
    def step(self, action):
        current_row = self.df.iloc[self.current_step]
        ask_price = current_row['ask_price_1']
        bid_price = current_row['bid_price_1']
        mid_price = current_row['mid_price']
        
        # Track previous value to calculate step reward
        prev_portfolio_val = self._calculate_portfolio_value(mid_price)
            
        # 1. Evaluate pending limit orders from previous step
        self._evaluate_limit_orders(ask_price, bid_price)
            
        # 2. Process new action from agent
        if action == 1: # Market BUY
            if self.position <= 0: 
                self._execute_trade(ask_price, direction=1, fee_rate=-self.taker_fee)
        elif action == 2: # Market SELL
            if self.position >= 0:
                self._execute_trade(bid_price, direction=-1, fee_rate=-self.taker_fee)
        elif action == 3: # Limit BUY (Post at current bid)
            if self.position <= 0:
                self.pending_limit_buy = bid_price
        elif action == 4: # Limit SELL (Post at current ask)
            if self.position >= 0:
                self.pending_limit_sell = ask_price
                
        # 3. Calculate new value and reward
        new_portfolio_val = self._calculate_portfolio_value(mid_price)
        reward = new_portfolio_val - prev_portfolio_val
                
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        info = {
            'balance': self.balance,
            'position': self.position,
            'portfolio_value': new_portfolio_val
        }
        
        return self._get_observation(), float(reward), bool(terminated), bool(truncated), info

    def _calculate_portfolio_value(self, current_mid_price):
        val = self.balance
        if self.position == 1:
            val += (current_mid_price - self.entry_price)
        elif self.position == -1:
            val += (self.entry_price - current_mid_price)
        return val

    def _execute_trade(self, price, direction, fee_rate):
        """
        Executes a trade, applying fees or rebates.
        fee_rate is negative for taker fee, positive for maker rebate.
        """
        if self.position == 0: # Entering position
            self.position = direction
            self.entry_price = price
            # Fee is relative to notional size of 1 BTC/unit
            self.balance += price * fee_rate 
        else: # Closing position (must be opposite direction)
            pnl = (price - self.entry_price) if self.position == 1 else (self.entry_price - price)
            self.balance += pnl + (price * fee_rate)
            self.position = 0

    def _evaluate_limit_orders(self, current_ask, current_bid):
        """
        Simulates filling of limit orders based on new order book state.
        A passive buy is filled if the new ask drops to or below the limit price (adverse selection).
        A passive sell is filled if the new bid rises to or above the limit price.
        """
        if self.pending_limit_buy is not None:
            if current_ask <= self.pending_limit_buy:
                self._execute_trade(self.pending_limit_buy, direction=1, fee_rate=self.maker_rebate)
            self.pending_limit_buy = None # Orders cancel if not filled immediately for simplicity

        if self.pending_limit_sell is not None:
            if current_bid >= self.pending_limit_sell:
                self._execute_trade(self.pending_limit_sell, direction=-1, fee_rate=self.maker_rebate)
            self.pending_limit_sell = None
