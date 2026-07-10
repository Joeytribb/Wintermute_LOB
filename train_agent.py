import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from rl_env import L2TradingEnv

def main():
    print("Initializing Custom Gym Environment...")
    # Path to our reconstructed features
    data_path = 'data/l2_features.csv'
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Ensure reconstruct_lob.py has finished running.")
        return

    # Create the environment
    # We use a window size of 10 to give the agent a sense of recent LOB momentum
    env_creator = lambda: L2TradingEnv(data_path, window_size=10, initial_balance=10000.0, taker_fee=0.0005)
    
    # Wrap it to normalize the extremely large numbers in the order book (amounts, prices)
    # This is CRITICAL for neural networks to converge
    env = DummyVecEnv([env_creator])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    print("Initializing PPO Agent...")
    # Initialize PPO
    # MlpPolicy is a standard feed-forward network. 
    # Because we flattened our 2D state into a 1D vector (handled implicitly by flattening the obs), MlpPolicy works well.
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048, batch_size=64, ent_coef=0.01, device='auto')
    
    print("Training Agent... (Targeting overnight 10M timesteps to reach 6% daily growth)")
    # Train for 10 Million timesteps
    total_timesteps = 10000000 
    model.learn(total_timesteps=total_timesteps)
    
    # Save the model and normalization stats
    model.save("ppo_l2_agent")
    env.save("vec_normalize.pkl")
    print("Training complete. Model saved.")
    
    print("Evaluating Agent...")
    # Evaluation run on the same environment (In a real scenario, evaluate on a holdout test set)
    env.training = False
    env.norm_reward = False
    
    obs = env.reset()
    balances = []
    
    # Run one full episode
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        balances.append(info[0]['balance'])
        
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(balances, label='RL Agent Portfolio Balance', color='blue')
    plt.title('PPO Agent Evaluation on BTCUSDT L2 Order Book')
    plt.xlabel('Time Step (1 step = 1 second)')
    plt.ylabel('Portfolio Balance (USD)')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(os.path.dirname(__file__), 'rl_evaluation.png')
    plt.savefig(plot_path)
    print(f"Evaluation complete. Plot saved to {plot_path}")

if __name__ == "__main__":
    main()
