import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

class L2TradingEnv(gym.Env):
    """
    Custom Environment for High-Frequency Trading on L2 Order Book data.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, data_path, window_size=10, initial_balance=10000.0, taker_fee=0.0005):
        super(L2TradingEnv, self).__init__()
        
        # Load the reconstructed order book features
        self.df = pd.read_csv(data_path)
        
        # Calculate mid_price for simple PnL tracking
        self.df['mid_price'] = (self.df['bid_price_1'] + self.df['ask_price_1']) / 2.0
        
        # Calculate Advanced Features
        bid_vol = self.df[[f'bid_amount_{i}' for i in range(1, 6)]].sum(axis=1)
        ask_vol = self.df[[f'ask_amount_{i}' for i in range(1, 6)]].sum(axis=1)
        self.df['obi'] = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-8)
        self.df['spread'] = self.df['ask_price_1'] - self.df['bid_price_1']
        
        # Drop timestamp for the feature vector
        self.features = self.df.drop(columns=['timestamp', 'mid_price']).values
        
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.taker_fee = taker_fee
        
        # Action space: 0 = Hold, 1 = Buy (Long), 2 = Sell (Short)
        self.action_space = spaces.Discrete(3)
        
        # Observation space: 20 LOB features * window_size
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
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Start at window_size to have enough history
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0.0
        
        return self._get_observation(), {}
        
    def _get_observation(self):
        # Return a rolling window of LOB features
        obs = self.features[self.current_step - self.window_size:self.current_step]
        # We could normalize here, but we will let PPO's VecNormalize handle it
        return obs.astype(np.float32)
        
    def step(self, action):
        current_row = self.df.iloc[self.current_step]
        ask_price = current_row['ask_price_1']
        bid_price = current_row['bid_price_1']
        mid_price = current_row['mid_price']
        
        reward = 0.0
        trade_occurred = False
        
        # Execution Logic
        if action == 1: # BUY
            if self.position == 0: # Enter Long
                self.position = 1
                self.entry_price = ask_price
                reward -= ask_price * self.taker_fee # Pay fee
                trade_occurred = True
            elif self.position == -1: # Close Short
                # Profit = (Entry - Exit)
                pnl = (self.entry_price - ask_price)
                reward += pnl - (ask_price * self.taker_fee)
                self.balance += reward
                self.position = 0
                trade_occurred = True
                
        elif action == 2: # SELL
            if self.position == 0: # Enter Short
                self.position = -1
                self.entry_price = bid_price
                reward -= bid_price * self.taker_fee # Pay fee
                trade_occurred = True
            elif self.position == 1: # Close Long
                # Profit = (Exit - Entry)
                pnl = (bid_price - self.entry_price)
                reward += pnl - (bid_price * self.taker_fee)
                self.balance += reward
                self.position = 0
                trade_occurred = True
                
        # If we didn't close a position, we still want to reward unrealized PnL to guide the agent
        # We calculate the delta in unrealized PnL from the last step to this step
        if not trade_occurred and self.position != 0:
            prev_row = self.df.iloc[self.current_step - 1]
            prev_mid = prev_row['mid_price']
            
            if self.position == 1:
                unrealized_delta = mid_price - prev_mid
                reward += unrealized_delta
            elif self.position == -1:
                unrealized_delta = prev_mid - mid_price
                reward += unrealized_delta
                
        # We could also apply a small penalty for holding a position (inventory risk)
        # or a penalty for doing nothing to encourage trading, but we'll stick to pure PnL for now.
                
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        info = {
            'balance': self.balance,
            'position': self.position
        }
        
        return self._get_observation(), reward, terminated, truncated, info
