# Architecture Diagram & System Flow

## High-Level Data Flow

```mermaid
graph TD
    A[Raw L2 Ticks (BTCUSDT)] -->|extract.py| B(CSV Tick Data)
    B -->|reconstruct_lob.py| C[In-Memory Order Book]
    C -->|1-second downsample| D[LOB Snapshots (Top 5 Bids/Asks)]
    D -->|Feature Engineering| E[State Vector: Prices, Vols, OBI, Spread]
    
    subgraph Reinforcement Learning Environment
        E --> F[L2TradingEnv - Gymnasium]
        F -->|State: Windowed Features| G[PPO Agent]
        G -->|Action: Buy/Sell/Hold| F
        F -->|Reward: PnL Step| G
    end
    
    G -->|train_agent.py| H[(ppo_l2_agent.zip)]
    F -->|VecNormalize| I[(vec_normalize.pkl)]
    
    H -->|eval.py / run_test.py| J[Out-of-Sample Evaluation]
    I --> J
    J --> K[Performance Metrics & Plots]
```

## Core Architecture Components

### 1. Data Reconstruction Layer (`reconstruct_lob.py`)
This module handles the transition from asynchronous, event-driven data (ticks) to synchronous, time-driven data (states). It maintains dynamic dictionaries of bids and asks. On passing a 1-second interval threshold, it sorts the dictionaries to isolate the Top 5 levels of the book and appends this state to an output matrix.

### 2. State & Reward Layer (`rl_env.py`)
The `L2TradingEnv` dictates the physical laws of the simulation:
*   **State Space:** A temporal sliding window (size=10) of 20 raw LOB features + Order Book Imbalance (OBI) + Spread. 
*   **Execution Logic:** Simulated market orders that interact exclusively with Level 1 (`ask_price_1` for buys, `bid_price_1` for sells).
*   **Reward Function:** The step-by-step change in portfolio value (Realized + Unrealized PnL).

### 3. RL Training Layer (`train_agent.py`)
Relies on Stable-Baselines3. Uses `DummyVecEnv` for synchronous environment stepping and `VecNormalize` to standardize observations dynamically (a necessity given the massive numerical scale of BTC prices and volumes). The model uses a standard feed-forward `MlpPolicy`.

### 4. Evaluation Layer (`run_test.py`)
Isolates a strictly unseen test dataset, rebuilds the environment, and loads the frozen `VecNormalize` statistics to prevent data leakage. Runs a deterministic pass of the policy to calculate final Out-Of-Sample (OOS) PnL.
