import os
import time
from collections import deque

import numpy as np
from tqdm import tqdm

import gymnasium
import homeostatic_robot_sim_env

env = gymnasium.make("OnTheBoard-v0")

print(env.action_space)
print(env.observation_space)

L = 50000

target_hist = deque(maxlen=L)
joint_hist = deque(maxlen=L)

print("### RESET")
env.reset()
steps = 0
done = False
action = np.ones_like(env.action_space.sample())

for _ in tqdm(range(L)):
    if steps % 100 == 0 and steps != 0:
        action[:] = 1 if action[0] == -1 else -1
        
    print(action)

    setpoint = env.wrapped_env.action2angle(action)
    target_hist.append(setpoint)
    
    obs, rew, done, truncated, info = env.step(action)
    done = done | truncated

    joint_hist.append(env.wrapped_env.get_joint_position())
    env.render()
    
    steps += 1

os.makedirs("homeostatic_robot_sim_env/data_pid", exist_ok=True)

import matplotlib.pyplot as plt

time_tick = np.arange(L) * env.dt
for i in range(8):
    plt.subplot(4, 2, i+1)
    plt.plot(time_tick, np.vstack(target_hist)[:, i], "k--", alpha=0.6)
    plt.plot(time_tick, np.vstack(joint_hist)[:, i], "r", alpha=0.6)
    
    np.save(f"homeostatic_robot_sim_env/data_pid/target{i}.npy", np.vstack(target_hist)[:, i])
    np.save(f"homeostatic_robot_sim_env/data_pid/joint{i}.npy", np.vstack(joint_hist)[:, i])
    
    if i == 0:
        plt.xlabel("time [sec]")
        plt.ylabel("angle [rad]")

np.save(f"homeostatic_robot_sim_env/data_pid/time_tick.npy", time_tick)

env.close()

plt.tight_layout()
plt.show()

print(f"Finish at {steps}")
