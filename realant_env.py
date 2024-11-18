import json
import time
from typing import Optional

import numpy as np
import zmq
import gymnasium
from gymnasium import spaces
from numpy._typing import NDArray

from const import TARGET_DURATION, KEYS_FROM_ROBOT, OBS_SCALING
from utils import action2servo, get_joint, sliceable_deque, qtoeuler

context = zmq.Context()

socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")


def encode_action_message(action: NDArray) -> bytes:
    m = action2servo(action)
    
    s = f"s1 {m[0]} s2 {m[1]} s3 {m[2]} s4 {m[3]} s5 {m[4]} s6 {m[5]} s7 {m[6]} s8 {m[7]}\n"
    s = bytes(s.encode("utf-8"))
    
    return s


class RealAntEnv(gymnasium.Env):
    def __init__(self,
                 obs_stack=1,
                 action_as_obs=False,  # set previous action as a component of obs
                 leg_obs_only=False,
                 joint_only=False,
                 no_joint_vel=False,
                 ):
        self.action_as_obs = action_as_obs
        self.leg_obs_only = leg_obs_only
        self.joint_only = joint_only
        self.no_joint_vel = no_joint_vel
        
        self.measurement_shape = 22
        if self.leg_obs_only:
            self.measurement_shape = 16
        if self.joint_only:
            self.measurement_shape = 8
        if self.no_joint_vel:
            self.measurement_shape = 8 + 6

        self.obs_shape = self.measurement_shape * obs_stack
        if self.action_as_obs:
            self.obs_shape += 8
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_shape,), dtype=np.float64)
        self._set_action_space()
        
        self.last_pos = None
        self.last_joint_positions = None
        self.obs_stack = obs_stack
        self.obs_dict_now = None
        
        self.past_measurements = sliceable_deque(maxlen=self.obs_stack)
        
        self.action_now = np.zeros_like(self.action_space.sample())
        
        self._step = 0

    def _set_action_space(self):
        self.action_space = spaces.Box(low=-np.ones(8), high=np.ones(8), dtype=np.float32)
        return self.action_space
    
    def reset(self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
):
        socket.send(b"reset")
        time.sleep(1)

        self.action_now = np.zeros_like(self.action_space.sample())
        
        obs_dict = json.loads(socket.recv())
        obs_dict.update({k: int(obs_dict[k]) / OBS_SCALING[k] for k in KEYS_FROM_ROBOT})

        self.last_pos = obs_dict["ANT"][0]
        self.obs_dict_now = obs_dict

        obs = self.reset_model(obs_dict)
        info = {}

        return obs, info
    
    def reset_model(self, obs_dict):
        self._step = 0
        
        self.past_measurements = sliceable_deque(
            [self.get_latest_measurement(obs_dict) for _ in range(self.obs_stack)],
            maxlen=self.obs_stack
        )
        
        return self.get_current_obs(obs_dict)
    
    def step(self, action: NDArray):
        action_ = np.clip(action, a_min=-1, a_max=1)
        
        # exponential filtering of motor actions
        self.action_now = self.action_now + 0.8 * (action_ - self.action_now)

        action_message = encode_action_message(self.action_now)

        # Step
        success = False
        while not success:
            socket.send(action_message)
            obs_dict = json.loads(socket.recv())
            try:
                obs_dict.update({k: int(obs_dict[k]) / OBS_SCALING[k] for k in KEYS_FROM_ROBOT})
                success = True
            except ValueError:
                pass
            
        self.obs_dict_now = obs_dict

        obs = self.get_current_obs(obs_dict)
        reward = self.get_reward(obs_dict)

        terminal = False
        truncated = False
        info = {}
        
        self._step += 1

        return obs, reward, terminal, truncated, info
    
    def get_current_obs(self, obs_dict):
        meas = self.get_latest_measurement(obs_dict)

        self.past_measurements.append(meas)
        
        meas = list(self.past_measurements)

        if self.action_as_obs:
            meas.append(self.action_now)
        
        return np.concatenate(meas, dtype=np.float32)
    
    def get_latest_measurement(self, obs_dict: dict) -> np.ndarray:
        #body_xyz = np.array([obs_dict["ANT"][0][2], obs_dict["ANT"][0][0], obs_dict["ANT"][0][1]])
        body_rpy_ = qtoeuler(obs_dict["ANT"][1])  # pitch, yaw, roll

        joint_positions = get_joint(obs_dict)

        if self.last_joint_positions is None:
            self.last_joint_positions = joint_positions
            
        joint_positions_vel = (joint_positions - self.last_joint_positions) / TARGET_DURATION  # approx. fixme
        
        if self.leg_obs_only:
            measurement = np.concatenate([
                joint_positions,
                joint_positions_vel,
            ], dtype=np.float32)
        elif self.joint_only:
            measurement = np.concatenate([
                joint_positions,
            ], dtype=np.float32)
        elif self.no_joint_vel:
            measurement = np.concatenate([
                np.sin(body_rpy_),
                np.cos(body_rpy_),
                joint_positions,
            ], dtype=np.float32)
        else:
            measurement = np.concatenate([
                np.sin(body_rpy_),
                np.cos(body_rpy_),
                joint_positions,
                joint_positions_vel,
            ], dtype=np.float32)
        
        return measurement
    
    def get_joint_position(self):
        return get_joint(self.obs_dict_now)
    
    def get_reward(self, obs_dict: dict):
        x, y, z = obs_dict["ANT"][0]
        
        speed = x - self.last_pos[0]
        
        self.last_pos = obs_dict["ANT"][0]

        return speed
    
    def close(self):
        socket.send(b"reset")
        
    def off_torque(self):
        socket.send(b"detach_servos")

    def on_torque(self):
        socket.send(b"attach_servos")


if __name__ == '__main__':
    env = RealAntEnv()
    
    env.reset()
    
    action = np.random.uniform(-1, 1, 8) * 0.3
    for _ in range(20):
        action[0] = 0
        action[1] = -1
        action[2] = 0
        action[3] = 1
        action[4] = 0
        action[5] = 1
        action[6] = 0
        action[7] = -1
        action += np.random.uniform(-1, 1, 8)

        obs, reward, terminal, truncated, info = env.step(action)
        
        print(obs, reward, terminal, truncated, info)
        
    env.close()
    
    print("Finish.")
