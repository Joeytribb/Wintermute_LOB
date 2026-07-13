import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

class L2TradingEnv(gym.Env):
    """
    Custom Environment for High-Frequency Trading on L2 Order Book data.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, data_path, window_size=10, initial_balance=10000.0, taker_fee=0.0005, start_idx=None, end_idx=None):
        super(L2TradingEnv, self).__init__()
        
        # Load the reconstructed order book features
        self.df = pd.read_csv(data_path)
        
        # Apply Train/Test split
        if start_idx is not None or end_idx is not None:
            self.df = self.df.iloc[start_idx:end_idx].reset_index(drop=True)
        
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
        
        # Calculate prev portfolio value
        prev_portfolio_val = self.balance
        if self.position == 1:
            prev_portfolio_val += (mid_price - self.entry_price)
        elif self.position == -1:
            prev_portfolio_val += (self.entry_price - mid_price)
            
        # Execution Logic
        if action == 1: # BUY
            if self.position == 0: # Enter Long
                self.position = 1
                self.entry_price = ask_price
                self.balance -= ask_price * self.taker_fee # Pay fee
            elif self.position == -1: # Close Short
                pnl = (self.entry_price - ask_price)
                self.balance += pnl - (ask_price * self.taker_fee)
                self.position = 0
                
        elif action == 2: # SELL
            if self.position == 0: # Enter Short
                self.position = -1
                self.entry_price = bid_price
                self.balance -= bid_price * self.taker_fee # Pay fee
            elif self.position == 1: # Close Long
                pnl = (bid_price - self.entry_price)
                self.balance += pnl - (bid_price * self.taker_fee)
                self.position = 0
                
        # Calculate new portfolio value
        new_portfolio_val = self.balance
        if self.position == 1:
            new_portfolio_val += (mid_price - self.entry_price)
        elif self.position == -1:
            new_portfolio_val += (self.entry_price - mid_price)
            
        # Reward is strictly the change in portfolio value
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
