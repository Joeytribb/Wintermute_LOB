# High-Frequency Market Making via Proximal Policy Optimization in Continuous L2 State Spaces

**Author:** [Your Name]  
**Repository:** `Wintermute_LOB`  
**Focus Area:** Deep Reinforcement Learning, Market Microstructure, Optimal Execution  

## 1. Abstract
This project presents a custom, continuous-state reinforcement learning environment designed to solve the optimal execution and market-making problem in cryptocurrency Limit Order Books (LOB). By reconstructing raw Level 2 (L2) tick data into a synchronous state space, we map the Top 5 Bids and Asks, inventory metrics, and unrealized PnL into a Markov Decision Process (MDP). A Proximal Policy Optimization (PPO) agent was trained on this environment. Preliminary out-of-sample evaluations demonstrate a 162.38% theoretical return, primarily achieved by navigating adverse selection, penalizing spread-crossing (taker fees), and successfully capturing maker rebates. 

## 2. Infrastructure & Environment Design

### 2.1 Order Book Reconstruction (`reconstruct_lob.py`)
Processing raw, asynchronous L2 tick updates sequentially during neural network training introduces severe computational bottlenecks. To solve this, a custom order book matching engine was implemented. The engine ingests raw bid/ask tick deltas and maintains a continuous, in-memory state of the LOB. This state is downsampled at fixed microsecond intervals to generate a standardized, highly dimensional state vector (Top 5 depth, price imbalances, and liquidity gradients).

### 2.2 The Markov Decision Process (`rl_env.py`)
A custom `gymnasium` environment bridges the LOB state vectors to the RL agent. 
- **State Space:** Continuous and normalized representation of LOB depth, current agent inventory (position limits), and floating PnL.
- **Action Space:** Discrete execution parameters (`Buy`, `Sell`, `Hold`).
- **Reward Shaping Mechanism:** The reward function mathematically enforces the physics of the exchange. The agent incurs immediate penalties for executing Market Orders (paying the 0.05% taker fee) and receives positive reinforcement for executing Limit Orders that provide liquidity (capturing the maker rebate), while simultaneously penalizing absolute inventory divergence to maintain a delta-neutral profile.

## 3. Agent Training & Optimization
The continuous state-space control problem was solved utilizing Proximal Policy Optimization (PPO), implemented via `Stable-Baselines3`. PPO was selected due to its sample efficiency and ability to maintain stable gradient updates within the highly stochastic, non-stationary environment of financial time-series data. 

## 4. Preliminary Results & Discussion
Out-of-sample forward testing of the trained PPO agent yielded the following performance metrics:
- **Initial Balance:** $10,000.00
- **Final Balance:** $26,237.94
- **Net Profit:** $16,237.94 (162.38%)

### 4.1 Academic Limitations & Future Research
While the theoretical out-of-sample returns demonstrate the model's ability to successfully converge and learn the mathematical relationships of order book imbalances, these results highlight the necessity for rigorous academic evaluation. 

In high-frequency quantitative environments, returns of this magnitude on solo independent projects strongly suggest the presence of latent microstructure biases. The primary objective for future research is to formally evaluate this architecture against:
1. **Lookahead Bias:** Ensuring zero future-state leakage during the microsecond state-vector downsampling.
2. **Execution Latency:** Introducing stochastic delay penalties into the `gymnasium` environment to model the physical limitations of network latency and adverse selection (queue position).
3. **Market Impact:** Modeling the degradation of the order book liquidity proportional to the agent's simulated execution volume.

## 5. Conclusion
This repository establishes the foundational data engineering and reinforcement learning infrastructure required to apply Deep RL to high-frequency limit order books. The next phase of this research requires a formal academic framework to stress-test the agent against real-world microstructure execution constraints.
