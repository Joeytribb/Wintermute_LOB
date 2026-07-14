# Deep Reinforcement Learning in Non-Stationary Microstructures

## 1. Abstract
This paper presents a robust mathematical framework for deploying Proximal Policy Optimization (PPO) algorithms within high-frequency Limit Order Books (LOB). By rejecting standard retail data science practices (such as raw price feeding and point-estimate holdout splits), we establish a methodology heavily influenced by institutional quantitative research. We explicitly constrain the Markov Decision Process (MDP) with deterministic execution friction and validate the agent’s edge using Deflated Sharpe Ratios and Walk-Forward Embargoed Cross-Validation.

## 2. The Stationarity Mandate
Neural networks act as universal function approximators, yet they routinely fail in financial markets due to non-stationarity. A model trained on a regime with $\mu_1, \sigma_1$ will catastrophically fail when the underlying statistical moments shift to $\mu_2, \sigma_2$.
To prevent this, the input state-space $S_t$ must be mathematically proven to be stationary (lacking a unit root).
*   **Methodology:** We utilized the Augmented Dickey-Fuller (ADF) test (`stationarity_test.py`).
*   **Results:** While raw Ask/Bid prices strictly fail the ADF test ($p > 0.05$), derived features such as Order Book Imbalance (OBI) and fractional Spread successfully reject the null hypothesis ($p < 0.01$). Thus, the PPO agent is exclusively fed these stationary, volatility-normalized inputs.

## 3. The Execution Friction Barrier
Theoretical RL agents often achieve massive profitability by exploiting zero-latency arbitrage. To tether the model to empirical reality, we engineered an execution penalty within the Gymnasium environment.
*   **The Reward Function ($R_t$):**
    $$ R_t = \Delta \text{Portfolio Value} - (\text{Price} \times \tau) $$
    where $\tau$ represents the institutional taker fee (e.g., 2 bps).
*   **Findings:** The agent successfully isolated structural L2 imbalances, generating +10.34% frictionless OOS return. However, empirical evaluation proves that this edge is entirely consumed at fee tiers $\ge 3$ bps, establishing a strict thermodynamic barrier to market entry.

## 4. Deflated Sharpe Ratio (DSR) & Multiple Testing Bias
A critical error in Deep RL is training an agent for 100,000 iterations and reporting the final Out-Of-Sample Sharpe Ratio. Due to the high number of trials (epochs/hyperparameters), the probability of finding a false-positive strategy (False Discovery Rate) approaches 1.
*   **Methodology:** We implemented the Deflated Sharpe Ratio (`dsr_metrics.py`). We calculate the Expected Maximum Sharpe Ratio of the random noise across $N$ trials and penalize the agent's observed Sharpe.
*   **Results:** The final Sharpe Ratio is deflated, outputting a probabilistic confidence interval. The agent must pass a $>95\%$ DSR confidence to be deemed statistically significant.

## 5. Walk-Forward Embargoed Cross-Validation
Standard $K$-Fold or 80/20 splits leak temporal data in finance. If $t_{train}$ precedes $t_{test}$, the serial correlation of the time-series allows the model to "peek" into the future.
*   **Methodology:** We implemented a Purged Walk-Forward Evaluator (`walk_forward_eval.py`). Between every Train and Test block, an *Embargo Window* (e.g., 2% of the dataset) is dropped. This breaks the serial correlation, strictly immunizing the Out-Of-Sample equity curve against look-ahead bias.
