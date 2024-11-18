import os
import time
from collections import deque
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from realant_env import RealAntEnv
from const import KEYS_JOINT, KEYS_TEMPERATURE, KEYS_BATTERY, FLAT_POSTURE_ACTION
from utils import get_joint, get_motor_temperature

np.set_printoptions(suppress=True, precision=3)

env = RealAntEnv()
env.reset()

L = 200

# Data
data_joint = deque([np.zeros(8)] * L, maxlen=L)
data_temp = deque([40 + np.zeros(8)] * L, maxlen=L)
data_battery = deque([np.zeros(5)] * L, maxlen=L)
data_fps = deque([5.] * L, maxlen=L)
time_tick = deque(maxlen=L)
start_at = time.time()


action = FLAT_POSTURE_ACTION.copy()

last_step_at = time.time()

# Initial plots


def plots():
    plt.clf()
    plt.subplot(2, 3, 1)
    plt.title("hip temp")
    plt.plot(np.vstack(data_temp)[:, ::2], alpha=0.7)
    plt.grid()

    plt.subplot(2, 3, 2)
    plt.title("ankle temp")
    plt.plot(np.vstack(data_temp)[:, 1::2], alpha=0.7)
    plt.grid()

    plt.subplot(2, 3, 3)
    plt.title("hip")
    plt.plot(np.vstack(data_joint)[:, ::2], alpha=0.7)
    plt.grid()

    plt.subplot(2, 3, 4)
    plt.title("ankle")
    plt.plot(np.vstack(data_joint)[:, 1::2], alpha=0.7)
    plt.grid()

    plt.subplot(2, 3, 5)
    plt.title("fps")
    plt.plot(np.array(data_fps), alpha=0.7)
    plt.grid()

    plt.subplot(2, 3, 6)
    plt.title("volt")
    plt.plot(np.vstack(data_battery)[:, 1])
    plt.grid()

    plt.tight_layout()
    plt.pause(0.0001)


while True:
    env.step(action)
    
    fps = 1./(time.time() - last_step_at)
    last_step_at = time.time()
    
    obs_dict = env.obs_dict_now
    
    # save data
    time_tick.append(time.time())
    data_joint.append(get_joint(obs_dict))
    data_temp.append(get_motor_temperature(obs_dict))
    data_battery.append(np.array([obs_dict[k] for k in KEYS_BATTERY]))
    data_fps.append(fps)
    
    plots()
    # else:
    #     p = int(50 * len(time_tick) / L)
    #     print("Data Collecting... ||" + "+" * p + "-" * (50 - p) + "|")
    
    # print(data_temp)
