import json
import os
import time

import numpy as np
import gymnasium as gym
import homeostatic_robot_sim_env
import torch
import torch.nn as nn

from const import FLAT_POSTURE_ACTION
from datetime import datetime
from model_utils import layer_init, BetaHead, LayerNormGELU
from real_homeostatic_robot_env import RealHomeostaticRobotEnv
from utils import get_joint

FILE = "saved_model/2023-10-11-avg-20runs/PlayroomBase-v2__ppo_cooling__9__1696487298/PlayroomBase-v2__ppo_cooling__9__1696487298_final.pth"  # 1

IS_SIM = True  # False for realant
SAVE_DATA = False
N = 100_000  # maximum steps

# Demonstration Mode. This mode use fixed energy level. and don't stop experiment when the food is moved
DEMO_MODE = False

# Debug/Demo Settings
MOTION_OFF = False  # debug setting. if True the motor output is kept at flat action
NO_FOOD = False  # debug setting. if True the food is not detected.
FOOD_COLLECTION_ENERGY = False  # demo mode setting. if True the robot is hungry. otherwise robot is full.
FIX_TEMP = False  # ddebug setting. if True the all normalized motor temperature are at setpoint

# Env constant. Don't change!
AVG_TEMP = True
DEFAULT_MOTOR_CTRL = 0.5 * (FLAT_POSTURE_ACTION + 1)


class Agent(nn.Module):
    def __init__(self, n_stack=3):
        super().__init__()

        # joint only by default
        self.obs_size = 8 * n_stack

        # no positional obs. action as obs and range finder
        self.obs_size += 8 + 20

        # temperature v3 model (energy + temp)
        if AVG_TEMP:
            self.obs_size += 1 + 1
        else:
            self.obs_size += 1 + 8

        # joint action plus cooling behavior
        self.action_size = 9

        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.obs_size, 256)),
            LayerNormGELU(256),
            layer_init(nn.Linear(256, 64)),
            LayerNormGELU(64),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        self.actor = nn.Sequential(
            layer_init(nn.Linear(self.obs_size, 256)),
            LayerNormGELU(256),
            layer_init(nn.Linear(256, 64)),
            LayerNormGELU(64),
            BetaHead(64, self.action_size),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        probs = self.actor(x)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def decode_action(self, action: np.ndarray, verbose=False):
        # decoding the action of the agent to the motor control

        prob_cooling = action[8]
        motor_action = action[:8]

        flag_cooling = np.random.rand() < prob_cooling
        if flag_cooling:
            motor_action = DEFAULT_MOTOR_CTRL

        if verbose:
            if flag_cooling:
                print("FLAT_ACTION")
            print("ACTION: ", motor_action)

        return motor_action


def make_new_env():
    if IS_SIM:
        new_env = gym.make(
            "HomeostaticRobotSim-v1",
            obs_delay=0,
            obs_stack=3,
            n_food=1,
            action_as_obs=True,
            joint_only=True,
            no_wall=False,  # <-- with wall
            realmode=True,
            random_position=False,
            internal_reset="full",  # "random",
            # show_sensor_range=True,
            # sensor_range=1,
            parametric_thermal_model=True,
            temp_max=False,
            thermal_model="v3",
            no_position_obs=True,
            # domain_randomization=True,
            show_sensor_range=False,
            average_temperature=AVG_TEMP,
        )
    else:
        new_env = RealHomeostaticRobotEnv(avg_temp=True, no_food=NO_FOOD)

    new_env = gym.wrappers.ClipAction(new_env)
    new_env = gym.wrappers.RescaleAction(new_env, 0, 1)  # for Beta policy
    return new_env


if SAVE_DATA:
    # num_data = len(glob.glob("data_playroom/data_*"))
    if DEMO_MODE:
        # postfix = "_demo"
        # postfix = "_cold_impulse"
        postfix = "_hot_impulse"
    else:
        postfix = ""

    file_path = f"data_playroom/data_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}" + postfix
    os.makedirs(file_path, exist_ok=True)

agent = Agent(n_stack=3)
agent.load_state_dict(torch.load(FILE, map_location="cpu"))
agent.eval()

# print(agent)

data = {
    "charge": [],
    "temp": [],
    "charge_normal": [],
    "temp_normal": [],
    "joint": [],
    "position": [],
    "time": [],
    "volt": [],
    "motor_action": [],
    "prob_cooling": [],
    "food_position": [],
    "is_food_captured": [],
}


def append_data(new_info, prob_cooling, motor_action):
    if not IS_SIM:
        data["charge"].append(new_info["charge"])
        data["temp"].append(new_info["temp"].tolist())
        data["charge_normal"].append(new_info["charge_normal"])
        data["temp_normal"].append(new_info["temp_normal"].tolist())
        data["joint"].append(get_joint(new_info).tolist())
        data["position"].append([new_info["ANT"][0][2], new_info["ANT"][0][0]])
        data["time"].append(time.time() - start_at)
        data["volt"].append(new_info["volt"])
        data["food_position"].append(new_info["food_pos"].tolist())
        data["is_food_captured"].append(new_info["is_food_captured"])
    data["motor_action"].append(motor_action.tolist())
    data["prob_cooling"].append(prob_cooling.tolist())


start_at = time.time()
save_at = time.time()

env = make_new_env()
obs, info = env.reset()
env.render()

if not IS_SIM and SAVE_DATA and not DEMO_MODE:
    print("Experiment Started. Place the food at the suggested place.")
    # os.system("say -v Alex Experiment Started. Place the food at the suggested place")
    input("[Replace Food] Press Enter if OK to start.")
    # os.system("say -v Alex Enter pressed. Start Experiment.")
else:
    # os.system("say -v Alex Start experiment immediately.")
    pass


def save_all_data():
    with open(file_path + f"/data_all.json", "w") as outfile:
        json.dump(data, outfile)


try:
    for step in range(N):
        obs_ = torch.Tensor(obs[None])
        if DEMO_MODE:
            if FOOD_COLLECTION_ENERGY:
                obs_[0, 52] = 0.5
                print("【DEMO MODE】Energy Level if fixed at: ", obs_[0, 52].item())
            else:
                obs_[0, 52] = 1.0
                print("【DEMO MODE】Energy Level if fixed at: ", obs_[0, 52].item())

        if FIX_TEMP:
            obs_[0, 53:] = 0

        if NO_FOOD:
            obs_[0, 32:52] = 0
            print("【NO FOOD OPTION】Range finder inputs are aALL ZERO NOW")

        # print("joints: ", obs_[0, :8])
        # print("finder: ", obs_[0, 32:52])
        # print("intero: ", obs_[0, 52:])

        with torch.no_grad():
            action = agent.get_action_and_value(obs_)[0].cpu().numpy()[0]
            motor_action = agent.decode_action(action)

        if MOTION_OFF:
            motor_action = DEFAULT_MOTOR_CTRL

        obs, reward, terminated, truncated, info = env.step(motor_action)

        env.render()
        append_data(info, prob_cooling=action[8], motor_action=motor_action)

        # print(f"motor_action: {motor_action}")
        # print(f"obs: {obs}")
        # print(f"ANT: {info['ANT']}, FOOD: {info['FOOD']}")
        print("prob-cooling: ", int(1000 * action[8]) / 1000)
        print("motor_action: ", (100 * motor_action).astype(int) / 100)

        if IS_SIM:
            print(
                f'CL: {int(1000 * env.env.get_interoception()[0]) / 1000}, NC: ---, VOLT: ---, TEMP: {(100 * env.env.thermal_model.get_temp_now()).astype(int) / 100}')
            # print(f'TEMP NORMAL: {[int(1000 * v) / 1000. for v in env.env.get_interoception()[1:]]}')
        else:
            print(
                f'CL: {int(1000 * info["charge_normal"]) / 1000}, NC: {info["charge"]}, VOLT: {info["volt"]}, TEMP: {[int(100 * v) / 100. for v in info["temp"]]}')
            # print(f'TEMP NORMAL: {[int(1000 * v)/1000. for v in info["temp_normal"]]}')

            if info["is_food_captured"] is True and not DEMO_MODE:
                print("Food Captured. Stop.")
                # os.system("say -v Alex food captured. press enter if okay.")
                if SAVE_DATA:
                    print("Data Saving...")
                    save_all_data()
                del env
                print("...DONE!")
                # input("[Change Battery] Press Enter if OK to restart.")

                # os.system("say -v Alex Enter Pressed. Create new environment.")
                env = make_new_env()
                obs, info = env.reset()
                env.render()
                # os.system("say -v Alex Place the food at the new suggested place")
                input("[Replace Food] Press Enter if OK to start.")
                # os.system("say -v Alex Enter Enter pressed. Resume experiment.")

        print(f"Time [sec]: {int(100 * (time.time() - start_at)) / 100}")
        print("---")

except KeyboardInterrupt:
    if SAVE_DATA:
        save_all_data()
        print("Data saved from Keyboard exception.")
finally:
    if SAVE_DATA:
        save_all_data()
        print("Data saved from the end of experiment.")

print("Finish. :)")
