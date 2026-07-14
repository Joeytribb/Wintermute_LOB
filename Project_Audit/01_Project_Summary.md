# Project Summary

## Objective
The repository `DeepRL_Microstructure` aims to apply Deep Reinforcement Learning (DRL) to cryptocurrency Level 2 (L2) Limit Order Book (LOB) data. The goal is to train an autonomous agent to execute optimal market making and directional scalping strategies by recognizing structural imbalances in the order book. 

## Problem Formulation
The market environment is framed as a Markov Decision Process (MDP) where the agent observes a temporal window of LOB states and takes discrete actions (Buy, Sell, Hold) to maximize its portfolio value. A core research question of the project is evaluating the "thermodynamic decay" of the learned alpha against centralized exchange execution friction (taker fees).

## Input Data
The pipeline ingests raw, asynchronous Level 2 incremental tick data (`BTCUSDT_L2.csv` and `BTCUSDT_L2_test.csv`), containing timestamped bid/ask updates, prices, and amounts.

## Output
The final outputs are a trained Proximal Policy Optimization (PPO) agent (`ppo_l2_agent.zip`), normalization statistics (`vec_normalize.pkl`), and performance plots (`rl_evaluation.png`, `rl_evaluation_test.png`) that map the agent's out-of-sample portfolio trajectory.

## Machine Learning Models & Algorithms
*   **Algorithm:** Proximal Policy Optimization (PPO), implemented via `Stable-Baselines3`.
*   **Policy Network:** Multi-Layer Perceptron (`MlpPolicy`), which maps flattened temporal LOB windows to discrete action probabilities and value estimates.

## Pipeline & Modules
1.  **Data Ingestion & Reconstruction (`extract.py`, `reconstruct_lob.py`):** Unzips raw ticks and sequentially processes them to maintain an in-memory limit order book. Downsamples the book at 1-second intervals to extract the Top 5 Bids/Asks (Price and Volume).
2.  **Environment (`rl_env.py`):** A custom `gymnasium` environment (`L2TradingEnv`) that calculates derived features (Order Book Imbalance, Spread), tracks simulated portfolio PnL, enforces taker fees, and processes order executions.
3.  **Training (`train_agent.py`):** Initializes the environment, wraps it in `VecNormalize` to standardize observations and rewards, and trains the PPO agent.
4.  **Evaluation (`eval.py`, `run_test.py`):** Evaluates the trained agent on out-of-sample data, applying identical normalization parameters to ensure consistency.

## Conclusion Reached by the Project
The author claims the agent successfully identifies structural imbalances in the LOB. However, empirical testing supposedly proves that the edge is entirely consumed by execution friction (taker fees $\ge 0.05\%$), rendering the strategy unprofitable for retail tiers but theoretically viable for zero-fee or institutional tiers.
