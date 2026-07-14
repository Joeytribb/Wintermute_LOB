# Code Review: Engineering & Architecture

A ruthless examination of the codebase reveals severe architectural flaws, hidden bugs, and performance bottlenecks that would disqualify this code from production environments in any tier-1 proprietary trading firm.

## 1. `reconstruct_lob.py`
*   **Performance Bottleneck / Memory Leak:** In `reconstruct_and_downsample`, `bids` and `asks` are maintained as Python dictionaries. Every 1 second, the code executes `sorted(bids.items())`. Because levels with `amount=0` are only removed if explicitly sent by the exchange, deeply out-of-the-money levels accumulate indefinitely. This results in $O(N \log N)$ sorting of massive, ever-growing dictionaries every single second.
*   **Missing Crossed-Book Handling:** There is no logic to detect or resolve crossed books (where bid > ask). In raw crypto L2 feeds, microsecond crossed books occur frequently. Failing to resolve these corrupts the state vector.
*   **Precision/Float Instability:** Using standard Python floats as dictionary keys (`price = float(row[6])`) is mathematically dangerous for price levels due to floating-point representation errors. Decimal types or integer ticks must be used.

## 2. `rl_env.py`
*   **Fatal Leverage / PnL Bug:** The agent starts with a `balance` of $10,000 (`initial_balance=10000.0`). However, when executing trades, the PnL is calculated based on the raw price difference of 1 full Bitcoin. E.g., `pnl = (bid_price - self.entry_price)`. If BTC moves from $60,000 to $61,000, the agent makes $1,000. This implies the agent is trading 1 full BTC (~$60k nominal) with only $10k in capital, representing an implicit 6x leverage. However, margin requirements and liquidation checks are entirely absent. 
*   **Fee Calculation Inconsistency:** 
    *   On Long Entry: `self.balance -= ask_price * self.taker_fee`
    *   On Short Close: `self.balance += pnl - (ask_price * self.taker_fee)`
    *   Wait, the `balance` is treated as unencumbered cash + realized PnL, but the capital required to purchase the asset is never subtracted. This means the agent trades with infinite capital and arbitrary position sizing (hardcoded to 1 unit).
*   **Faulty Position Flipping:** `action == 1` (Buy) will close a Short position, leaving the agent Flat (`self.position = 0`). To go from Short to Long, the agent would need to take Action 1 *twice in a row*. This is an unintuitive and restrictive action space. 

## 3. `train_agent.py` & `eval.py`
*   **Hardcoded Magic Numbers:** The split index `end_idx=69037` in `train_agent.py` and `start_idx=69037` in `eval.py` is hardcoded. If the underlying data changes length, this will break silently or cause train/test overlap.
*   **Dead Code / Config:** `window_size=10` is passed to the environment, but the state flattening logic assumes the model handles it gracefully. 
*   **Short Training Loop:** `total_timesteps = 100,000` is grossly insufficient for a PPO agent learning on highly noisy financial time series. Given a dataset of 69,037 rows, 100k timesteps is roughly 1.4 epochs. The model is merely memorizing the local drift of the first 24 hours of data.
