# Documentation & API Reference

*Note: The current repository lacks significant docstrings. This document provides the professional standard documentation that should be implemented.*

## Module: `reconstruct_lob`

### `reconstruct_and_downsample(input_file: str, output_file: str, interval_us: int = 1000000)`
Parses asynchronous Level 2 incremental tick data and reconstructs a synchronous, time-series Limit Order Book (LOB).
*   **Parameters:**
    *   `input_file` (str): Path to the raw CSV tick data.
    *   `output_file` (str): Destination path for the reconstructed state vectors.
    *   `interval_us` (int): Downsampling interval in microseconds (default 1,000,000 for 1 second).
*   **Output:** Writes a CSV file containing timestamped snapshots of the Top 5 Bids and Asks (Prices and Amounts).

## Module: `rl_env`

### `class L2TradingEnv(gym.Env)`
A custom Gymnasium environment simulating execution physics on a historical Limit Order Book.

#### `__init__(self, data_path: str, window_size: int = 10, initial_balance: float = 10000.0, taker_fee: float = 0.0005, start_idx: int = None, end_idx: int = None)`
Initializes the trading environment.
*   **Parameters:**
    *   `data_path`: Path to reconstructed LOB features.
    *   `window_size`: Number of temporal steps included in the state observation.
    *   `initial_balance`: Starting portfolio cash balance.
    *   `taker_fee`: Percentage fee charged on market executions (0.0005 = 5 bps).
    *   `start_idx`, `end_idx`: Row indices for train/test splitting.

#### `reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]`
Resets the environment state, clearing inventory and reverting balance to `initial_balance`.
*   **Returns:** Initial state observation (shape: `window_size` x `n_features`).

#### `step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]`
Executes an action in the market and transitions to the next state.
*   **Action Space:** `0` (Hold), `1` (Buy/Close Short), `2` (Sell/Close Long).
*   **Returns:** `obs` (next state), `reward` (step portfolio PnL), `terminated`, `truncated`, `info` (dict containing balance and position).

## Module: `train_agent` / `eval`

### `main()`
Entry points for training and evaluating the Stable-Baselines3 PPO agent.
*   `train_agent.py`: Initializes the environment (fee = 0.0), wraps with `VecNormalize`, trains for 100k steps, and saves the model.
*   `eval.py` / `run_test.py`: Loads the frozen `VecNormalize` parameters, runs a deterministic evaluation of the agent on unseen data, and outputs portfolio trajectory plots.
