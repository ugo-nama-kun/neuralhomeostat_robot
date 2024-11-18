import numpy as np

from realant_env import RealAntEnv

env = RealAntEnv()
env.reset()

for _ in range(20):
    action = np.random.uniform(-1, 1, 8) * 0
    # action[0] = np.random.uniform(-1, 1)
    
    obs, reward, terminal, truncated, info = env.step(action)
    
    print(obs, reward, terminal, truncated, info)

env.close()

print("Finish.")
