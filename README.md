# Deep Reinforcement Learning on High-Frequency Limit Order Books

## Abstract
This repository implements an empirical framework for applying Deep Reinforcement Learning (DRL) to Level 2 (L2) Limit Order Books. The primary objective is to train an autonomous agent to perform optimal market making and directional scalping by observing the microstructure of the order book. While theoretical models often yield significant alpha in frictionless environments, this research also explicitly maps the thermodynamic decay of that edge against centralized exchange taker fees. The agent successfully identifies structural imbalance in the LOB (+10.34% OOS frictionless return) but empirically proves that profitability requires sub-3 bps institutional execution tiers or DeFi mempool latency advantages.

## Mathematical Formulation & Architecture

### 1. Market State Reconstruction (`reconstruct_lob.py`)
To map the continuous, stochastic flow of L2 incremental ticks into a Markov Decision Process (MDP), the order book is reconstructed into standardized state vectors.
*   **State Space ($\mathcal{S}$):** Normalized representation of the Top 5 Bids/Asks (Price and Volume), generating a continuous depth vector that captures the instantaneous supply/demand imbalance.

### 2. Reinforcement Learning Environment (`rl_env.py`)
A custom `gymnasium` environment models the execution physics:
*   **Action Space ($\mathcal{A}$):** Discrete market orders (`Buy`, `Sell`, `Hold`).
*   **Reward Function ($\mathcal{R}$):** The agent is explicitly penalized for crossing the spread (taker fees $\tau$) and rewarded for closing positions at a net profit. 
*   **Execution Friction:** The environment rigorously enforces slippage and exchange fees on every state transition, preventing the neural network from exploiting zero-latency/zero-fee arbitrage anomalies that do not exist in reality.

### 3. Deep RL Agent (`train_agent.py`)
Utilizes Proximal Policy Optimization (PPO) via `Stable-Baselines3`. PPO was selected for its robust clipping mechanism, ensuring stable gradient updates in highly stochastic, non-stationary financial time series.

## Empirical Results: The Friction Barrier

Out-Of-Sample (OOS) evaluation of the trained PPO agent demonstrates a linear decay in predictive yield relative to taker fees:

| Taker Fee ($\tau$) | OOS Return | Final Balance | Conclusion |
| :--- | :--- | :--- | :--- |
| **0.000% (0 bps)** | +10.34% | $11,034.46 | Theoretical Maximum |
| **0.010% (1 bps)** | +7.37% | $10,737.12 | Institutional Tier (Profitable) |
| **0.020% (2 bps)** | +4.40% | $10,439.78 | Institutional Tier (Profitable) |
| **0.030% (3 bps)** | +1.42% | $10,142.45 | Threshold limit |
| **0.050% (5 bps)** | -4.52% | $9,547.78 | Retail Tier (Edge Destroyed) |

**Conclusion:** The neural network successfully identifies a statistically significant edge in L2 structural imbalances. However, execution friction acts as a thermodynamic barrier; the alpha is entirely consumed by taker fees $\ge 0.05\%$. 

## Running the Pipeline
```bash
# 1. Reconstruct the L2 tick data into state vectors
python reconstruct_lob.py

# 2. Train the PPO agent (Requires tuning for specific fee parameters)
python train_agent.py

# 3. Evaluate Out-of-Sample performance across fee tiers
python eval.py
```
