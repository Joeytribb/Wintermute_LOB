import pandas as pd
import numpy as np
import os

def generate_data():
    os.makedirs('data', exist_ok=True)
    n_steps = 50000
    
    # 1. Non-stationary raw price (random walk)
    np.random.seed(42)
    price_changes = np.random.normal(0, 1, n_steps)
    ask_price = 60000 + np.cumsum(price_changes)
    
    # 2. Stationary features (mean-reverting)
    # Order Book Imbalance (OBI) oscillates around 0
    obi = np.random.normal(0, 0.5, n_steps)
    # Autoregressive component to make it look realistic
    for i in range(1, n_steps):
        obi[i] = 0.8 * obi[i-1] + np.random.normal(0, 0.2)
        
    # Spread is strictly mean reverting around 1.0
    spread = np.clip(np.random.normal(1.0, 0.2, n_steps), 0.5, 5.0)
    for i in range(1, n_steps):
        spread[i] = 0.5 * spread[i-1] + 0.5 + np.random.normal(0, 0.1)
        
    df = pd.DataFrame({
        'ask_price_1': ask_price,
        'obi': obi,
        'spread': spread
    })
    
    df.to_csv('data/l2_features.csv', index=False)
    print("Realistic L2 data generated.")

if __name__ == '__main__':
    generate_data()
