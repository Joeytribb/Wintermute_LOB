import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import seaborn as sns
import numpy as np
import os

sns.set_theme(style="whitegrid", context="paper")

def plot_stationarity():
    df = pd.read_csv('data/l2_features.csv')
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=300)
    fig.suptitle('Markov Decision Process State-Space Stationarity Proofs', fontsize=16, fontweight='bold')
    
    # 1. Non-Stationary Raw Price
    price = df['ask_price_1'][:1000]
    axes[0, 0].plot(price, color='#e74c3c')
    axes[0, 0].set_title('Non-Stationary Input: Raw Price (Rejected)', fontsize=12)
    axes[0, 0].set_ylabel('Price (USD)')
    
    # 2. Stationary OBI
    obi = df['obi'][:1000]
    axes[0, 1].plot(obi, color='#2ecc71')
    axes[0, 1].set_title('Stationary Input: Order Book Imbalance (OBI)', fontsize=12)
    axes[0, 1].set_ylabel('Imbalance Ratio')
    
    # 3. ACF of Price (Shows no rapid decay -> Non-Stationary)
    plot_acf(price, ax=axes[1, 0], lags=50, title='ACF: Raw Price', color='#e74c3c')
    
    # 4. ACF of OBI (Shows rapid decay -> Stationary)
    plot_acf(obi, ax=axes[1, 1], lags=50, title='ACF: Order Book Imbalance', color='#2ecc71')
    
    plt.tight_layout()
    plt.savefig('stationarity_plot.png')
    print("Saved stationarity_plot.png")

if __name__ == '__main__':
    plot_stationarity()
