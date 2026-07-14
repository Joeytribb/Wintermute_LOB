# Quantitative Finance & Trading Review

If submitted as a portfolio piece to a tier-1 quant fund (Jane Street, Citadel, Two Sigma, Oxford-Man), this repository would fail the technical screen immediately due to fundamental misunderstandings of market microstructure.

## 1. Zero Market Impact & Infinite Liquidity
`rl_env.py` assumes the agent can execute a full 1 BTC market order (approx. $60,000 nominal) identically at `ask_price_1` or `bid_price_1`. In reality, top-of-book (L1) liquidity on crypto exchanges is often thin. A $60k market order will sweep the book, incurring significant slippage (crossing L2, L3, etc.). By assuming infinite L1 liquidity, the simulation overstates returns drastically. 

## 2. Unrealistic Latency Assumptions
The Gym environment executes the agent's action instantaneously at the observed state's price. In HFT, processing the L2 tick, passing it through a neural network, and routing the order to the exchange takes milliseconds. By the time the order arrives, the structural imbalance the agent identified will have been arbitraged away by faster participants (adverse selection). The environment must include stochastic latency delays.

## 3. Leverage & Position Sizing Illusions
As noted in the Code Review, the agent starts with $10,000 but trades 1 full unit of BTC. This is 6x leverage, but the environment does not charge margin interest, does not restrict buying power, and has no liquidation engine. The agent is effectively trading with infinite, free capital. 

## 4. Missing Risk Metrics
A PnL curve is meaningless to a quant fund without risk adjustment. The evaluation scripts do not calculate:
*   **Sharpe / Sortino Ratio:** How much risk was taken to achieve the return?
*   **Maximum Drawdown:** Did the agent lose 50% of its capital before recovering?
*   **Win Rate & Profit Factor:** Is the agent relying on a few lucky trades, or a consistent statistical edge?
*   **Information Ratio:** Performance relative to a benchmark.

## 5. Thermodynamic Barrier Misunderstanding
The README notes that returns degrade linearly as taker fees increase. This is physically obvious, but the conclusion ("alpha is entirely consumed by taker fees") is a symptom of poor execution strategy, not a fundamental law. The agent uses purely aggressive market orders. Real high-frequency market making relies on passive limit orders (capturing the spread and maker rebates) and only crosses the spread when structural alpha exceeds the taker fee. Because the environment has no limit order capabilities, the "thermodynamic barrier" is artificially constructed by the author's restricted action space.
