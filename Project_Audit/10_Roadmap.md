# Refactoring Roadmap

This roadmap is prioritized from critical mathematical flaws that invalidate the system, to technical debt that slows down research.

## Priority 1: Critical Research & ML Fixes (Immediate)
*   **Stationarize Features:** Modify `reconstruct_lob.py` or `rl_env.py` to convert absolute prices to relative distances from the mid-price (e.g., `spread_bps`, `distance_to_mid`). 
*   **Fix POMDP State Space:** Add `self.position` and `floating_pnl` (normalized) to the observation vector in `rl_env.py`.
*   **Fix Train/Test MDP Mismatch:** Train separate models for each `taker_fee` tier, or include `taker_fee` as an observable feature so a single agent can learn fee-conditional policies.
*   **Fix the PnL/Margin Bug:** Implement proper position sizing. If balance is $10k, position size should be fractional BTC (e.g., 0.1 BTC). Subtract the total nominal value from the balance upon entry, or implement explicit margin/leverage math.

## Priority 2: High Priority Architecture
*   **Neural Architecture:** Replace `MlpPolicy` with a recurrent policy (`RecurrentPPO` or a custom LSTM/CNN feature extractor) to properly process the `window_size=10` sequential data.
*   **Rewrite README & Summary:** Delete false claims about "maker rebates" and "162% returns". Standardize the documentation to reflect reality.
*   **Extend Training:** Increase `total_timesteps` from 100k to at least 5-10 million to ensure policy convergence. 

## Priority 3: Medium Priority Engineering Debt
*   **Optimize Order Book Engine:** Replace the Python dictionary sorting in `reconstruct_lob.py` with a Red-Black Tree, B-Tree, or simply maintain a sorted list using `bisect`. Clear levels where `amount == 0` properly, and implement logic to handle crossed books.
*   **Remove Hardcoded Indices:** Implement dynamic train/test splitting based on percentages (e.g., 80/20) and pass them cleanly via configuration files (e.g., YAML/Hydra).

## Priority 4: Low Priority / "Nice to Have"
*   **Add Limit Orders:** Expand the action space to allow placing passive limit orders at various depths, and model queue position (latency).
*   **Risk Metrics:** Calculate Sharpe, Sortino, and Max Drawdown in `eval.py`.
*   **Experiment Tracking:** Integrate Weights & Biases (`wandb`) for tracking training curves and hyperparameter tuning.
