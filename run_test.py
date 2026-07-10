import gzip
import shutil
import os
import time
from reconstruct_lob import reconstruct_and_downsample
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from rl_env import L2TradingEnv
import matplotlib.pyplot as plt

def main():
    test_gz = 'data/BTCUSDT_L2_test.csv.gz'
    test_csv = 'data/BTCUSDT_L2_test.csv'
    test_features = 'data/l2_features_test.csv'
    
    print("1. Extracting out-of-sample data...")
    if not os.path.exists(test_csv):
        with gzip.open(test_gz, 'rb') as f_in:
            with open(test_csv, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    print("Extraction complete!")
    
    print("2. Reconstructing LOB and generating features...")
    if not os.path.exists(test_features):
        reconstruct_and_downsample(
            input_file=test_csv,
            output_file=test_features,
            interval_us=1000000 # 1 second
        )
    print("Reconstruction complete!")
    
    print("3. Evaluating Agent on OUT OF SAMPLE DATA...")
    env_creator = lambda: L2TradingEnv(test_features, window_size=10, initial_balance=10000.0, taker_fee=0.0005)
    env = DummyVecEnv([env_creator])
    
    # Load normalization stats from training, DO NOT UPDATE THEM during testing
    env = VecNormalize.load('vec_normalize.pkl', env)
    env.training = False
    env.norm_reward = False
    
    model = PPO.load('ppo_l2_agent', env=env)
    obs = env.reset()
    
    balances = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, info = env.step(action)
        balances.append(info[0]['balance'])
        
    final_balance = info[0]['balance']
    pnl = final_balance - 10000.0
    perc = (pnl / 10000.0) * 100
    
    print("\n--- OUT OF SAMPLE RESULTS ---")
    print(f"INITIAL BALANCE: $10000.00")
    print(f"FINAL BALANCE: ${final_balance:.2f}")
    print(f"NET PROFIT: ${pnl:.2f} ({perc:.2f}%)")
    print("-----------------------------\n")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(balances, label='RL Agent (Out of Sample)', color='green')
    plt.title('PPO Agent Evaluation on OUT OF SAMPLE Data (2023-02-01)')
    plt.xlabel('Time Step (1 step = 1 second)')
    plt.ylabel('Portfolio Balance (USD)')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(os.path.dirname(__file__), 'rl_evaluation_test.png')
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

if __name__ == '__main__':
    main()
