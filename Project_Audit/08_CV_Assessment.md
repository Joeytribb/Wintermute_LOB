# Curriculum Vitae Assessment

Assume this repository is submitted as a supplementary portfolio piece for a PhD application to the Oxford-Man Institute or a Quant Researcher role at a Tier-1 firm.

## Scoring (Out of 10)

| Category | Score | Justification |
| :--- | :--- | :--- |
| **Novelty** | 3/10 | Vanilla PPO applied to basic tabular LOB features. Heavily explored in literature. |
| **Difficulty** | 5/10 | Building a custom Gym environment and LOB reconstructor requires solid engineering, but the implementation relies heavily on standard libraries without optimizing the math/logic. |
| **Engineering Quality** | 2/10 | O(N log N) sorting every second, memory leaks, hardcoded indices, raw script execution (no argparse/config). |
| **Research Quality** | 1/10 | Contradictory claims in the README/Summary. False claims about Maker rebates. Look-ahead biases. Train/Test MDP mismatch. |
| **Code Quality** | 3/10 | Readable, but lacks docstrings, typing, modularity, and error handling. |
| **Reproducibility** | 6/10 | Scripts execute linearly, but hardcoded array indices and lack of explicit package dependencies (`requirements.txt`) dock points. |
| **Scientific Rigor** | 2/10 | No risk metrics, no baselines, no multiple seed evaluation, raw non-stationary prices used in neural networks. |
| **Overall Impact** | 3/10 | A superficial implementation of Deep RL in finance. It demonstrates the applicant knows how to import `stable_baselines3`, but proves they do not fundamentally understand machine learning (stationarity, POMDPs) or market microstructure (latency, margin, impact). |

## CV Value
**Verdict: Negative Value.**
In its current state, including this repository on a CV for an elite program will actively harm the applicant's chances. Any rigorous technical reviewer will immediately spot the train/test fee mismatch, the leverage bug, and the non-stationarity of the price inputs. It suggests a "copy-paste" approach to data science rather than first-principles quantitative research.
