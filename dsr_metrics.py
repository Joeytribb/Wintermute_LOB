import numpy as np
import scipy.stats as stats

def calculate_deflated_sharpe_ratio(returns, num_trials, variance_of_trials):
    """
    Calculates the Deflated Sharpe Ratio (DSR) as proposed by Bailey & Lopez de Prado.
    Penalizes the final Out-Of-Sample Sharpe Ratio based on how many hyperparameters/epochs
    were tested (Multiple Testing Bias).
    
    Parameters:
    - returns: Array of daily/step out-of-sample returns.
    - num_trials: Number of models/epochs evaluated (e.g., 100 for hyperparameter tuning).
    - variance_of_trials: The variance of the Sharpe Ratios across all trials.
    
    Returns:
    - Estimated Deflated Sharpe Ratio (Probability that the True Sharpe > 0)
    """
    
    # 1. Calculate Standard Sharpe
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    
    if std_ret == 0:
        return 0, 0
        
    # Annualization factor (assuming steps are seconds, 365 days)
    ann_factor = np.sqrt(365 * 24 * 60 * 60)
    sharpe = (mean_ret / std_ret) * ann_factor
    
    # 2. Calculate the Expected Maximum Sharpe Ratio (False Discovery Rate)
    # Using the approximation for the expected maximum of independent normal variables
    euler_mascheroni = 0.5772
    
    # The expected maximum Sharpe ratio under the null hypothesis (due to random chance)
    expected_max_sr = np.sqrt(variance_of_trials) * (
        (1 - euler_mascheroni) * stats.norm.ppf(1 - 1/num_trials) +
        euler_mascheroni * stats.norm.ppf(1 - 1/(num_trials * np.e))
    )
    
    # 3. Calculate DSR (Deflated Sharpe Ratio)
    # This is the probability that our observed Sharpe ratio is statistically significantly 
    # greater than the Expected Maximum Sharpe Ratio generated purely by multiple-testing noise.
    
    T = len(returns)
    skewness = stats.skew(returns)
    kurtosis = stats.kurtosis(returns)
    
    # Asymptotic variance of the Sharpe Ratio
    sr_variance = (1 - (skewness * sharpe) + ((kurtosis - 1)/4) * (sharpe**2)) / (T - 1)
    
    if sr_variance <= 0:
        dsr_prob = 0.0
    else:
        # Z-score of our Sharpe relative to the expected maximum noise
        z_score = (sharpe - expected_max_sr) / np.sqrt(sr_variance)
        dsr_prob = stats.norm.cdf(z_score)
        
    return sharpe, dsr_prob

def main():
    print("--- Deflated Sharpe Ratio (DSR) Analysis ---")
    print("Evaluating the risk of Multiple Testing Bias in Deep RL.")
    
    # Simulated Out-Of-Sample returns from the PPO agent
    np.random.seed(42)
    # Agent makes tiny consistent returns, with occasional heavy tail losses
    simulated_returns = np.random.normal(0.00001, 0.001, 20000) 
    
    # We trained for 100,000 steps, let's assume we evaluated 50 different model checkpoints
    num_trials = 50 
    var_trials = 0.5 # Variance of Sharpe ratios across those 50 checkpoints
    
    sharpe, dsr_prob = calculate_deflated_sharpe_ratio(simulated_returns, num_trials, var_trials)
    
    print(f"\nObserved Annualized Sharpe Ratio: {sharpe:.4f}")
    print(f"Number of Models Evaluated: {num_trials}")
    print(f"\nDeflated Sharpe Ratio (Probability True Sharpe > 0): {dsr_prob * 100:.2f}%")
    
    if dsr_prob > 0.95:
        print("=> CONCLUSION: The alpha is statistically significant (Passes 95% DSR Confidence).")
    else:
        print("=> CONCLUSION: The alpha is a statistical illusion caused by Multiple Testing Overfit.")

if __name__ == '__main__':
    main()
