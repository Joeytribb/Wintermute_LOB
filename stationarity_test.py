import os
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

def test_stationarity(timeseries, feature_name):
    """
    Perform Augmented Dickey-Fuller (ADF) test on a time series.
    H0 (Null Hypothesis): The time series is non-stationary (has a unit root).
    H1 (Alt Hypothesis): The time series is stationary.
    """
    print(f"\n--- ADF Test for {feature_name} ---")
    
    # Drop NaNs
    ts = timeseries.dropna()
    
    if len(ts) < 100:
        print("Not enough data to test stationarity.")
        return
        
    result = adfuller(ts, autolag='AIC')
    
    adf_statistic = result[0]
    p_value = result[1]
    critical_values = result[4]
    
    print(f"ADF Statistic: {adf_statistic:.4f}")
    print(f"p-value: {p_value:.10f}")
    print("Critical Values:")
    for key, value in critical_values.items():
        print(f"  {key}: {value:.4f}")
        
    if p_value < 0.05:
        print("=> REJECT Null Hypothesis: The feature is STRICTLY STATIONARY.")
    else:
        print("=> ACCEPT Null Hypothesis: The feature is NON-STATIONARY (Unit root present).")

def main():
    data_path = 'data/l2_features.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run reconstruct_lob.py first.")
        return
        
    df = pd.read_csv(data_path)
    print(f"Loaded dataset with {len(df)} ticks.")
    
    # We must prove that our state-space features are stationary.
    # We test Order Book Imbalance (obi) and Spread.
    
    if 'obi' in df.columns:
        test_stationarity(df['obi'], 'Order Book Imbalance (OFI)')
        
    if 'spread' in df.columns:
        test_stationarity(df['spread'], 'Bid-Ask Spread')
        
    # We also test raw price to prove that raw price is NON-STATIONARY
    if 'ask_price_1' in df.columns:
        test_stationarity(df['ask_price_1'], 'Raw Ask Price (Should Fail)')

if __name__ == '__main__':
    main()
