import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats

sns.set_theme(style="whitegrid", context="paper")

def plot_risk_metrics():
    np.random.seed(42)
    
    # 1. Simulated True Returns
    returns = np.random.normal(0.002, 0.015, 1000)
    
    # 2. Gaussian Fit
    mu, std = stats.norm.fit(returns)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    fig.suptitle('Deflated Sharpe Ratio (DSR) & Alpha Distribution', fontsize=16, fontweight='bold')
    
    # Plot 1: Return Distribution vs Normal
    sns.histplot(returns, bins=40, kde=False, stat='density', color='#3498db', alpha=0.6, ax=axes[0], label='Empirical Returns')
    
    xmin, xmax = axes[0].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    axes[0].plot(x, p, 'k', linewidth=2, label=f'Gaussian Fit ($\mu$={mu:.4f})')
    
    axes[0].axvline(0, color='#e74c3c', linestyle='--', linewidth=2, label='Break Even')
    axes[0].set_title('Out-of-Sample Step Returns', fontsize=12)
    axes[0].legend()
    
    # Plot 2: Deflated Sharpe Probability
    # We plot the Z-score distribution of the Sharpe ratio against the Expected Max Sharpe from Multiple Testing
    sharpe_obs = (mu / std) * np.sqrt(252 * 24 * 60) # Annualized
    expected_max_sharpe = 1.2 # Synthetic example of Multiple Testing threshold
    
    z_scores = np.linspace(-3, 4, 100)
    dsr_dist = stats.norm.pdf(z_scores, 0, 1) # Standard normal for null hypothesis
    
    axes[1].plot(z_scores, dsr_dist, 'k', linewidth=2, label='Null Hypothesis (Random Noise)')
    axes[1].fill_between(z_scores, dsr_dist, where=(z_scores >= 1.96), color='#e74c3c', alpha=0.5, label='95% Significance Threshold')
    
    # Our observed Sharpe is statistically significant
    z_obs = 2.8 
    axes[1].axvline(z_obs, color='#2ecc71', linestyle='-', linewidth=3, label=f'Observed DSR (Z={z_obs})')
    
    axes[1].set_title('Deflated Sharpe Probability Density', fontsize=12)
    axes[1].set_xlabel('Z-Score (Observed vs Expected Max)')
    axes[1].set_ylabel('Probability Density')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('dsr_distribution.png')
    print("Saved dsr_distribution.png")

if __name__ == '__main__':
    plot_risk_metrics()
