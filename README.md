# Deep Reinforcement Learning on High-Frequency Limit Order Books (LOB)

## Project Overview
This repository implements a high-frequency trading (HFT) reinforcement learning environment based on Level 2 (L2) Limit Order Book data. The goal is to train an autonomous agent to perform optimal market making and directional scalping by observing the microstructure of the order book (Top 5 Bids/Asks and depth imbalances).

In latency-sensitive environments, traditional indicator-based strategies fail due to adverse selection. This agent learns to identify micro-imbalances in the LOB and execute trades before the spread shifts.

## Architecture

### 1. LOB Reconstruction (`reconstruct_lob.py`)
Raw L2 tick data (incremental updates) is computationally expensive to process sequentially during RL training. This module acts as an Order Book Matching Engine:
- Ingests raw bid/ask tick deltas.
- Maintains a continuous state of the order book in memory.
- Downsamples the book into standardized state vectors (e.g., Top 5 Bid/Ask prices and volumes) at fixed microsecond intervals (e.g., 1-second snapshots).

### 2. Reinforcement Learning Environment (`rl_env.py`)
A custom `gymnasium` environment that maps the LOB state vector into a Markov Decision Process (MDP):
- **State Space**: The continuous normalized representation of the Top 5 Bids and Asks, plus the agent's current inventory (position) and unrealized PnL.
- **Action Space**: Discrete actions (`Buy`, `Sell`, `Hold`). 
- **Reward Shaping**: The agent is heavily penalized for crossing the spread (paying taker fees) and rewarded for capturing the spread (maker rebates) and maintaining delta-neutral inventory limits.

### 3. PPO Agent (`train_agent.py`)
Utilizes Proximal Policy Optimization (PPO) via Stable-Baselines3 to solve the continuous state-space control problem. PPO was selected for its sample efficiency and stable gradient updates in highly stochastic financial environments.

## Running the Pipeline
```bash
# 1. Reconstruct the L2 tick data into state vectors
python reconstruct_lob.py

# 2. Train the PPO agent
python train_agent.py

# 3. Evaluate Out-of-Sample performance
python eval.py
```
