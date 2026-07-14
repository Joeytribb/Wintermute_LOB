import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper")

def plot_wfo():
    # Simulate WFO equity curve
    np.random.seed(42)
    
    n_folds = 4
    fold_size = 500
    embargo_size = 50
    
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    
    current_idx = 0
    equity = 10000.0
    
    colors = ['#3498db', '#9b59b6', '#f1c40f', '#e67e22']
    
    for i in range(n_folds):
        # Simulate returns for this OOS fold
        returns = np.random.normal(0.001, 0.005, fold_size)
        fold_equity = [equity]
        for r in returns:
            fold_equity.append(fold_equity[-1] * (1 + r))
            
        fold_equity = np.array(fold_equity)
        
        # Plot OOS Block
        x = np.arange(current_idx, current_idx + len(fold_equity))
        ax.plot(x, fold_equity, label=f'OOS Fold {i+1}', color=colors[i], linewidth=2)
        
        equity = fold_equity[-1]
        current_idx += len(fold_equity)
        
        # Plot Embargo Gap (except after last fold)
        if i < n_folds - 1:
            ax.axvspan(current_idx, current_idx + embargo_size, color='#e74c3c', alpha=0.3, label='Purged Embargo Window' if i==0 else "")
            current_idx += embargo_size
            
    ax.set_title('Combinatorial Purged Walk-Forward Optimization (Out-of-Sample Equity)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Trading Steps (Time)')
    ax.set_ylabel('Portfolio Value ($)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('wfo_equity.png')
    print("Saved wfo_equity.png")

if __name__ == '__main__':
    plot_wfo()
