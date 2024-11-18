import inspect
import math
import os
import pathlib
import collections
import itertools
import random
import tempfile
import xml.etree.ElementTree as ET

from typing import Optional

import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from gymnasium.vector.utils import spaces

from scipy.spatial.transform import Rotation as R

from playroom_env.envs.command_env import CommandEnv
from playroom_env.envs.on_the_board_env import OnTheBoardEnv
from playroom_env.envs.playroom_env2 import PlayroomEnv2

# Code is mostly based on the original paper and provided code
# https://github.com/alexlioralexli/rllab-finetuning/blob/2dae9141d0fdc284d04f18931907131d66b43023/sandbox/finetuning/envs/mujoco/ant_env.py
# and https://github.com/AaltoVision/realant-rl/blob/main/realant_sim/mujoco.py

DEFAULT_CAMERA_CONFIG = {
    "distance": 5.0,
}
FLAT_POSTURE_ACTION = np.array([0, -1, 0, 1, 0, 1, 0, -1], dtype=np.float32)


class sliceable_deque(collections.deque):
    # from https://stackoverflow.com/questions/10003143/how-to-slice-a-deque
    def __getitem__(self, index):
        if isinstance(index, slice):
            return type(self)(itertools.islice(self, index.start,
                                               index.stop, index.step))
        return collections.deque.__getitem__(self, index)


def q_inv(a):
    return [a[0], -a[1], -a[2], -a[3]]


def q_mult(a, b):  # multiply two quaternion
    w = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
    i = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2]
    j = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1]
    k = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]
    return [w, i, j, k]


def lerp(a: float, b: float, t: float) -> float:
    # Linear interpolate [-1, 1] -> [a, b]
    t_ = 0.5 * (t + 1)
    return (1 - t_) * a + t_ * b


import matplotlib.pyplot as plt
from collections import deque


class RealAntBaseEnv(MujocoEnv, utils.EzPickle):
    FILE = "realant_base.xml"  # xml originally from https://github.com/AaltoVision/realant-rl
    ORI_IND = 3
    IS_WALKER = True
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps"  : 20,
    }
    
    def __init__(self,
                 xml_path="realant_base.xml",
                 ego_obs=True,
                 xyz_noise_std=0.05,
                 rpy_noise_std=0.01,
                 joint_noise_std=0.01,
                 obs_delay=0,
                 obs_stack=1,
                 action_as_obs=False,  # set previous action as a component of obs
                 vision=False,
                 width=64,
                 height=64,
                 max_episode_steps=1_000,
                 domain_randomization=False,
                 leg_obs_only=False,
                 joint_only=False,
                 no_joint_vel=False,
                 no_wall=True,
                 activity_range=0.75,  # default wall activity range
                 *args, **kwargs):
                
        assert obs_stack > 0
        
        self.ego_obs = ego_obs
        self.max_episode_length = max_episode_steps
        self.action_as_obs = action_as_obs
        self.domain_randomization = domain_randomization
        self.leg_obs_only = leg_obs_only
        self.joint_only = joint_only
        self.no_joint_vel = no_joint_vel
        self.v3 = False if kwargs.get("v3") is not True else True
        self.no_wall = no_wall
        self.activity_range = activity_range
        
        self.frame_skip = 5
        self.width = width
        self.height = height
        self.vision = vision
        
        self.obs_delay = obs_delay  # 1 step = 50 ms
        self.obs_stack = obs_stack
        
        self.xyz_noise_std = xyz_noise_std
        self.rpy_noise_std = rpy_noise_std
        self.joint_noise_std = joint_noise_std
        
        self.int_err, self.past_err = 0, 0
        self.prev_data = deque(maxlen=2)
        self.past_measurements = sliceable_deque(maxlen=self.obs_stack + self.obs_delay)
        
        utils.EzPickle.__init__(self)
        
        self.measurement_shape = 22
        if self.leg_obs_only:
            self.measurement_shape = 16
        if self.joint_only:
            self.measurement_shape = 8
        if self.no_joint_vel:
            self.measurement_shape = 8 + 6
        
        self.obs_shape = self.measurement_shape * self.obs_stack
        if self.action_as_obs:
            self.obs_shape += 8
            
        # Build Once
        p = pathlib.Path(inspect.getfile(self.__class__))
        self.MODEL_DIR = os.path.join(p.parent, "models", xml_path)
        
        self.build_model()
        
        self._set_action_space()
        
        self.action_now = np.zeros_like(self.action_space.sample())
        
        self._step = 0
    
    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=-np.ones_like(low[::2]), high=np.ones_like(high[::2]), dtype=np.float32)
        return self.action_space
    
    def action2angle(self, action: np.ndarray, is_v3=False):
        joint_angles = np.zeros_like(action)
        joint_angles[0] = lerp(-0.69, 0.69, action[0])  # hip_1
        joint_angles[1] = lerp(0, 1.05, action[1]) # ankle_1
        joint_angles[2] = lerp(-0.69, 0.69, action[2])  # hip_2
        joint_angles[3] = lerp(-1.05, 0, -action[3] if is_v3 else action[3])  # ankle_2
        joint_angles[4] = lerp(-0.69, 0.69, action[4])  # hip_3
        joint_angles[5] = lerp(-1.05, 0, -action[5] if is_v3 else action[5])  # ankle_3
        joint_angles[6] = lerp(-0.69, 0.69, action[6])  # hip_4
        joint_angles[7] = lerp(0, 1.05, action[7])  # ankle_4
        return joint_angles
    
    def do_simulation(self, ctrl, n_frames):
        self._step_mujoco_simulation(ctrl, n_frames)
    
    def step(self, action):
        
        # exponential filtering of motor actions
        self.action_now = self.action_now + 0.8 * (action.copy() - self.action_now)

        ctrl = np.zeros(16)
        ctrl[::2] = self.action2angle(self.action_now, is_v3=self.v3)  # avoid velocity controller in mujoco
        
        prev_body_xyz = self.data.qpos.flat.copy()[:3]
        
        self.do_simulation(ctrl, self.frame_skip)
        
        # observation
        obs = self.get_current_obs()
        
        # walking reward
        body_xyz = self.data.qpos.flat.copy()[:3]
        body_xyz_vel = body_xyz - prev_body_xyz
        reward = body_xyz_vel[0]
        
        # direction reward
        direction_cost = -np.abs(self.get_ori())
        reward += 0.01 * direction_cost
        
        # action cost
        lb, ub = self.action_space.low, self.action_space.high
        scaling = (ub - lb) * 0.5
        ctrl_cost = -.5 * np.square((action - FLAT_POSTURE_ACTION) / scaling).sum()
        reward += - 0.001 * ctrl_cost
        
        state = self.state_vector()
        notdone = np.isfinite(state).all()
        done = not notdone
        
        self._step += 1
        terminated = done or self._step >= self.max_episode_length
        
        return obs, reward, terminated, False, {}
        
    def get_ori(self):
        ori = [0, 1, 0, 0]
        rot = self.data.qpos[self.__class__.ORI_IND:self.__class__.ORI_IND + 4]  # take the quaternion
        ori = q_mult(q_mult(rot, ori), q_inv(rot))[1:3]  # project onto x-y plane
        ori = math.atan2(ori[1], ori[0])
        return ori
    
    def get_joint_position(self):
        return self.data.qpos.flat.copy()[-8:]
    
    def get_noisy_data(self):
        body_xyz = self.data.qpos.flat.copy()[:3]
        body_quat = self.data.qpos.flat.copy()[3:7]
        body_rpy = R.from_quat(body_quat).as_euler('xyz')
        joint_positions = self.data.qpos.flat.copy()[-8:]
        joint_positions_vel = self.data.qvel.flat.copy()[-8:]
        
        # add noise
        body_xyz += np.random.randn(3) * self.xyz_noise_std
        body_rpy += np.random.randn(3) * self.rpy_noise_std
        joint_positions += np.random.randn(joint_positions.size) * self.joint_noise_std
        joint_positions_vel += np.random.randn(joint_positions_vel.size) * self.joint_noise_std
        
        return body_xyz, body_rpy, joint_positions, joint_positions_vel
    
    def get_latest_measurement(self):
        body_xyz, body_rpy, joint_positions, joint_positions_vel = self.get_noisy_data()
        
        body_rpy_ = body_rpy.copy()  # pitch, yaw, roll
        
        # print("rpy: ", body_rpy_)
        
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
        
        # if self.action_as_obs:
        #     measurement = np.concatenate([
        #         measurement,
        #         self.action_now,  # add current normalized motor targets as obs.
        #     ])
        
        return measurement
    
    def get_current_obs(self):
        latest_measurement = self.get_latest_measurement()
        
        self.past_measurements.append(latest_measurement)
        
        meas = list(self.past_measurements[self.obs_delay:])
        
        if self.action_as_obs:
            meas.append(self.action_now)
        
        return np.concatenate(meas, dtype=np.float32)
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        # save current initialization
        self.init_qpos_ = self.data.qpos.ravel().copy()
        self.init_qvel_ = self.data.qvel.ravel().copy()
        
        self._reset_simulation()

        self.action_now = np.zeros_like(self.action_space.sample())

        if self.domain_randomization:
            self.build_model()
        
        self.init_qpos = self.init_qpos_
        self.init_qvel = self.init_qvel_

        self.reset_model()

        if self.render_mode == "human":
            self.render()
        return self.get_current_obs(), {}
    
    def reset_model(self):
        self._step = 0
        self.int_err = 0
        self.past_err = 0
        
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.0001, high=0.0001)
        qpos[-8:] = qpos[-8:] + self.np_random.uniform(size=8, low=-.1, high=.1)
        qvel = self.init_qvel  # + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        
        self.past_measurements = sliceable_deque(
            [self.get_latest_measurement() for _ in range(self.obs_delay + self.obs_stack)],
            maxlen=self.obs_delay + self.obs_stack
        )
        
        return self.get_current_obs()
    
    def update_xml(self, tree, is_domain_rand: bool, no_wall=True, activity_range=0.75):
        worldbody = tree.find(".//worldbody")
        asset = tree.find(".//asset")
        
        ET.SubElement(
            asset, "texture", dict(
                name="floor_texture",
                type="2d",
                file=os.path.dirname(__file__) + "/models/texture/floor.png",
                width="1",
                height="1",
            )
        )
        ET.SubElement(
            asset, "material", dict(
                name="grass",
                texture="floor_texture",
                texrepeat="10 10",
            )
        )
        
        asset.find("texture").set("builtin", "none")
        asset.find("texture").set("file", os.path.dirname(__file__) + "/models/texture/room.png")
        asset.find("texture").set("gridsize", "3 4")
        asset.find("texture").set("gridlayout", ".U..LFRB.D..")
        
        worldbody.find("geom").set("material", "grass")
        
        if not no_wall:
            attrs = dict(
                type="box", conaffinity="1", condim="3", rgba="0 0 0 1"
            )
            
            walldist = 1.2 * activity_range
            
            ET.SubElement(
                worldbody, "geom", dict(
                    attrs,
                    name="wall1",
                    pos="0 -%f 0.25" % walldist,
                    size="%f 0.1 0.5" % walldist))
            ET.SubElement(
                worldbody, "geom", dict(
                    attrs,
                    name="wall2",
                    pos="0 %f 0.25" % walldist,
                    size="%f 0.1 0.5" % walldist))
            ET.SubElement(
                worldbody, "geom", dict(
                    attrs,
                    name="wall3",
                    pos="-%f 0 0.25" % walldist,
                    size="0.1 %f 0.5" % walldist))
            ET.SubElement(
                worldbody, "geom", dict(
                    attrs,
                    name="wall4",
                    pos="%f 0 0.25" % walldist,
                    size="0.1 %f 0.5" % walldist))
        
        if is_domain_rand:
            # random terrain
            size_ = np.random.randint(2, 9)  # hfield size from 2 to 8
            # height_ = 0.1 * np.random.rand()
            height_ = 0.01 * np.random.rand()
            ET.SubElement(
                asset, "hfield", dict(
                    name="hill",
                    file=os.path.dirname(__file__) + "/models/hill_height.png",
                    size=f"{size_} {size_} {height_} 0.1",
                )
            )
            
            # Random friction
            default_elem = tree.find(".//default").find(".//geom")
            # default_elem.set("friction", f"{random.uniform(0.5, 2.0)} 0.5 0.5")
            default_elem.set("friction", f"{random.uniform(1.3, 1.7)} 0.5 0.5")

            # Randomly attaching weights on the agent
            torso = tree.find(".//body[@name='torso']")
            
            new_body = ET.SubElement(
                torso, "body", dict(
                    name="random_weight",
                    pos="0 0 0",
                )
            )
            
            x, y, z = self.generate_random_point_in_body()
            attached_weight = random.uniform(0, 0.2)
            ET.SubElement(
                new_body, "geom", dict(
                    name=f"random_weight",
                    type="sphere",
                    size=f"{0.01}",
                    pos=f"{x} {y} {z}",
                    rgba="0 1 0 1",
                    mass=f"{attached_weight}"  # 0 to 0.2 kg random weight
                )
            )
            
            # subtraction of attached weight (inertial effect only)
            # base_elem = tree.find(".//geom[@name='base']")
            # new_body_weight = float(base_elem.get('mass')) - attached_weight
            # base_elem.set("mass", f"{new_body_weight}")
            
            # print(f"body weights: {new_body_weight} + {attached_weight} = {new_body_weight + attached_weight}")
        else:
            asset = tree.find(".//asset")
            ET.SubElement(
                asset, "hfield", dict(
                    name="hill",
                    file=os.path.dirname(__file__) + "/models/hill_height.png",
                    size=f"5 5 0.001 0.1",
                )
            )

        return tree

    def build_model(self):
        tree = ET.parse(self.MODEL_DIR)
        tree = self.update_xml(tree, is_domain_rand=self.domain_randomization, no_wall=self.no_wall, activity_range=self.activity_range)

        with tempfile.NamedTemporaryFile(mode='wt', suffix=".xml") as tmpfile:
            file_path = tmpfile.name
            tree.write(file_path)
            
            if hasattr(self, "mujoco_renderer"):
                self.mujoco_renderer.close()
            
            observation_space = Box(low=-np.inf, high=np.inf, shape=(self.obs_shape,), dtype=np.float64)
            
            MujocoEnv.__init__(
                self,
                file_path,
                self.frame_skip,
                observation_space,
                width=self.width if self.vision else 480,
                height=self.height if self.vision else 480,
                default_camera_config=DEFAULT_CAMERA_CONFIG
            )
    
    def generate_random_point_in_body(self):
        x = np.random.uniform(-0.04, 0.04)
        y = np.random.uniform(-0.04, 0.04)
        z = np.random.uniform(-0.015, 0.015)
        return x, y, z
    
    def render(
            self,
            mode='human',
            camera_id=None,
            camera_name=None
    ):
        return self.get_image(mode=mode, camera_id=camera_id, camera_name=camera_name)
    
    def get_image(
            self,
            mode='human',
            camera_id=None,
            camera_name=None
    ):

        im = self.mujoco_renderer.render(
            mode,
            camera_id,
            camera_name,
        )
        
        return im


class RealAntBasePlayroomEnv2(PlayroomEnv2):
    MODEL_CLASS = RealAntBaseEnv
    ORI_IND = 3
    
    def __init__(self,
                 obs_delay=0,
                 obs_stack=1,
                 action_as_obs=False,
                 domain_randomization=False,
                 average_temperature=False,
                 energy_only=False,
                 *args, **kwargs):
        super().__init__(
            obs_delay=obs_delay,
            obs_stack=obs_stack,
            action_as_obs=action_as_obs,
            domain_randomization=domain_randomization,
            average_temperature=average_temperature,
            energy_only=energy_only,
            *args,
            **kwargs
        )


class RealAntBaseOnTheBoardEnv(OnTheBoardEnv):
    MODEL_CLASS = RealAntBaseEnv
    ORI_IND = 3
    
    def __init__(self,
                 obs_delay=0,
                 obs_stack=1,
                 action_as_obs=False,
                 domain_randomization=False,
                 *args, **kwargs):
        super().__init__(
            obs_delay=obs_delay,
            obs_stack=obs_stack,
            action_as_obs=action_as_obs,
            domain_randomization=domain_randomization,
            *args,
            **kwargs
        )


class RealAntBaseCommandEnv(CommandEnv):
    MODEL_CLASS = RealAntBaseEnv
    ORI_IND = 3
    
    def __init__(self,
                 fixed_command=None,
                 obs_delay=0,
                 obs_stack=1,
                 action_as_obs=False,
                 domain_randomization=False,
                 *args, **kwargs):
        super().__init__(
            fixed_command=fixed_command,
            obs_delay=obs_delay,
            obs_stack=obs_stack,
            action_as_obs=action_as_obs,
            domain_randomization=domain_randomization,
            *args,
            **kwargs
        )
