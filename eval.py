from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from rl_env import L2TradingEnv

def main():
    # Evaluate Out-Of-Sample (OOS) using start_idx=69037 to hit the final 20%
    env_creator = lambda: L2TradingEnv('data/l2_features.csv', window_size=10, initial_balance=10000.0, taker_fee=0.0, start_idx=69037)
    env = DummyVecEnv([env_creator])
    env = VecNormalize.load('vec_normalize.pkl', env)
    env.training = False
    env.norm_reward = False
    
    model = PPO.load('ppo_l2_agent', env=env)
    obs = env.reset()
    
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, info = env.step(action)
    
    final_balance = info[0]['balance']
    print(f"INITIAL BALANCE: $10000.00")
    print(f"FINAL BALANCE: ${final_balance:.2f}")
    
    pnl = final_balance - 10000.0
    perc = (pnl / 10000.0) * 100
    print(f"NET PROFIT: ${pnl:.2f} ({perc:.2f}%)")

if __name__ == '__main__':
    main()
