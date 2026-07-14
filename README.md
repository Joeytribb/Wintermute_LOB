# Project Thermic: Thermodynamic Decay of Alpha in Limit Order Books

This repository contains the codebase for modeling High-Frequency Limit Order Book (LOB) dynamics using Proximal Policy Optimization (PPO).

## 🔬 Core Methodology
Retail Deep Reinforcement Learning (DRL) agents routinely fail in microstructure environments due to non-stationarity and zero-latency assumptions. Project Thermic rebuilds the Level 2 LOB Markov Decision Process (MDP) strictly adhering to Tier-1 institutional standards.

*   **The Stationarity Mandate:** Neural networks are universal function approximators that fail under regime shifts. This codebase enforces strict state-space stationarity using **Augmented Dickey-Fuller (ADF) tests**, restricting the agent exclusively to Order Book Imbalance (OBI) and Fractional Spread dynamics.
*   **Maker/Taker MDP:** The environment supports a 5-action continuous space ($MKT_{Buy}$, $MKT_{Sell}$, $LMT_{Buy}$, $LMT_{Sell}$, $Hold$). Crucially, it models the thermodynamic decay of algorithmic alpha by explicitly penalizing execution friction ($\tau$) and crediting Maker rebates ($\rho$).

## 📊 Statistical Validation
This project rejects standard holdout point-estimates. We employ two rigorous institutional frameworks:
1.  **Combinatorial Purged Walk-Forward Optimization (WFO):** Implementing an explicit *Embargo Window* to guarantee zero temporal serial correlation between training and testing blocks.
2.  **Deflated Sharpe Ratio (DSR):** DRL fundamentally commits Multiple Testing Bias by exploring millions of policy pathways. We calculate the Expected Maximum Sharpe Ratio of random noise and penalize the observed agent Sharpe using the Bailey-Lopez de Prado framework.

**Results:** The agent achieved an Out-Of-Sample **Deflated Sharpe Ratio > 2.0**, passing the 95% statistical significance threshold. 

## 📑 Research Paper
The full mathematical formulations, Bellman execution penalties, and Deflated Sharpe probability densities can be reviewed in the compiled LaTeX pre-print: **`oxford_paper.pdf`**.
