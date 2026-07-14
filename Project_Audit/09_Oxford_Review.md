# Oxford-Man Institute: Mock Interview & Review

*Persona: Professor at the Oxford-Man Institute of Quantitative Finance reviewing the applicant's portfolio.*

## Initial Impressions
"The applicant has attempted a highly ambitious intersection of Deep RL and Market Microstructure. The initiative to reconstruct L2 data manually is appreciated. However, as I review the mathematical formulation of the MDP, I see massive violations of both stochastic control theory and basic financial logic."

## Interview Questions I Would Ask You
If you made it to the interview stage, I would ask the following questions to test if you realize the flaws in your own work:

1. **On Stationarity:** "I see you are feeding raw BTC prices (e.g., $60,000) directly into an MLP and normalizing them with a running mean. What happens to your policy's output if the price of Bitcoin doubles out-of-sample? How does your neural network handle inputs that are +20 standard deviations outside its training distribution?"
2. **On Partial Observability:** "Can you point me to where in your state vector the agent observes its current inventory? If it cannot see its position, how does it mathematically differentiate between opening a new speculative long versus unwinding an existing short to manage risk?"
3. **On the Train/Test Mismatch:** "In your README, you present a table showing the decay of alpha against taker fees. But in your code, the agent was trained in a frictionless environment (0 fees). Do you believe it is academically valid to evaluate a policy in an MDP that operates under different transition dynamics than the one it was trained on?"
4. **On Market Mechanics:** "Your reward function calculates PnL based on the price of 1 full Bitcoin. With a starting balance of $10,000, how did you model margin requirements, and why didn't the agent face margin calls when the position moved against it?"

## What Would Concern Me
The gap between the claims in `RESEARCH_SUMMARY.md` ("capturing the maker rebate") and the actual codebase (market orders only). This suggests either profound sloppiness or an attempt to inflate the research without doing the engineering. In academia, this is fatal.

## How to Impress Me
To make this outstanding, you must:
1. Difference all prices to be relative to the mid-price.
2. Incorporate an inventory penalty into the reward function (e.g., Avellaneda-Stoikov style).
3. Switch from an MLP to an RNN/CNN architecture for the temporal LOB windows.
4. Model latency and L1 liquidity properly.
