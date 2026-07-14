# Machine Learning Review

The machine learning methodology contains critical conceptual errors that invalidate the purported results. If submitted to a top AI lab or quant fund, these flaws would result in immediate rejection.

## 1. Feature Engineering: Non-Stationarity & Normalization Failure
*   **The Flaw:** `self.features` in `rl_env.py` includes raw `bid_price_1`, `ask_price_1`, etc. (e.g., $60,000). The `train_agent.py` script relies on `VecNormalize` to compute a running mean and variance to standardize these prices.
*   **Why it's fatal:** Prices are non-stationary random walks. If the training data contains BTC prices oscillating around $60,000, `VecNormalize` freezes those statistics. If the out-of-sample test data (`run_test.py`) sees BTC drift to $70,000, the normalized inputs to the Neural Network will be massive (e.g., +15 standard deviations). The MLP's activation functions will saturate (or extrapolate wildly), leading to garbage action distributions.
*   **The Fix:** Prices must be made relative. Features should be differenced or expressed relative to the mid-price (e.g., `bid_price_i - mid_price`). 

## 2. Partial Observability (POMDP) Misspecification
*   **The Flaw:** The agent's Observation Space (`self._get_observation()`) returns ONLY the windowed LOB features. It **completely omits** the agent's current inventory (`self.position`) and entry price/floating PnL. 
*   **Why it's fatal:** How can an agent learn to close a position optimally if it doesn't know it holds one? The Markov property is violated. The agent is flying blind, rendering any learned policy purely reactive to market momentum rather than functioning as an inventory-aware market maker. (Note: `RESEARCH_SUMMARY.md` claims inventory is in the state space, but the code proves this is a hallucination).

## 3. Environment Mismatch (Train vs. Test)
*   **The Flaw:** In `train_agent.py`, the environment is initialized with `taker_fee=0.0`. The PPO agent learns a high-frequency policy tailored to a frictionless environment. In `run_test.py`, the agent is evaluated with `taker_fee=0.0005`.
*   **Why it's fatal:** The agent's learned Q-values and policy distributions are calibrated for an MDP where crossing the spread costs $0. Testing this frozen policy in an MDP where crossing the spread costs 5 bps guarantees catastrophic failure. The model will overtrade and bleed to death. If you want to test the impact of fees, the agent must be *trained* with those fees, or the fee must be an observable state variable.

## 4. Architecture Selection
*   **The Flaw:** `train_agent.py` uses `MlpPolicy` (a standard Feed-Forward Neural Network) on a flattened 2D temporal window `(window_size, n_features)`.
*   **Why it's fatal:** MLPs lack spatial/temporal translation invariance. The first step of the window is treated as an entirely different feature from the last step. A 1D Convolutional Neural Network (CNN) or an LSTM/GRU is strictly required to capture temporal dynamics in LOB data.

## 5. Reward Shaping & Noise
*   **The Flaw:** The reward is `new_portfolio_val - prev_portfolio_val`, which incorporates *unrealized* PnL based on the `mid_price`. 
*   **Why it's fatal:** Unrealized PnL fluctuates purely due to noise in the spread, even when the agent takes no action. This injects massive variance into the reward signal, making it difficult for PPO's advantage estimator to assign credit to the actual execution actions. Rewards should be sparse (realized upon closing) or based on execution advantage.
