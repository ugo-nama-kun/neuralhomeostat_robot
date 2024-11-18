import math
import os
import xml.etree.ElementTree as ET
import inspect
from collections import deque
from enum import Enum, auto

import glfw
import numpy as np
from PIL import Image

import mujoco

from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium import utils
from scipy.spatial.transform import Rotation

from playroom_env.envs.thermal_module_parametric3 import ThermalModuleParametric3

BIG = 1e6
DEFAULT_CAMERA_CONFIG = {}
FLAT_POSTURE_ACTION = np.array([0, -1, 0, 1, 0, 1, 0, -1], dtype=np.float32)
DRIVE_WEIGHTS = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)  # weighting of drives.

def euler2mat(euler):
    r = Rotation.from_euler('xyz', euler, degrees=False)
    return r.as_matrix()


class ObjectClass(Enum):
    FOOD = auto()


class InteroClass(Enum):
    ENERGY = auto()
    TEMPERATURE = auto()


def qtoeuler(q):
    """ quaternion to Euler angle

    :param q: quaternion
    :return:
    """
    phi = math.atan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
    theta = math.asin(2 * (q[0] * q[2] - q[3] * q[1]))
    psi = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    return np.array([phi, theta, psi])


def eulertoq(euler):
    phi, theta, psi = euler
    qx = np.cos(phi / 2) * np.cos(theta / 2) * np.cos(psi / 2) + np.sin(phi / 2) * np.sin(theta / 2) * np.sin(psi / 2)
    qy = np.sin(phi / 2) * np.cos(theta / 2) * np.cos(psi / 2) - np.cos(phi / 2) * np.sin(theta / 2) * np.sin(psi / 2)
    qz = np.cos(phi / 2) * np.sin(theta / 2) * np.cos(psi / 2) + np.sin(phi / 2) * np.cos(theta / 2) * np.sin(psi / 2)
    qw = np.cos(phi / 2) * np.cos(theta / 2) * np.sin(psi / 2) - np.sin(phi / 2) * np.sin(theta / 2) * np.cos(psi / 2)
    return np.array([qx, qy, qz, qw])


class PlayroomEnv2(MujocoEnv, utils.EzPickle):
    MODEL_CLASS = None
    ORI_IND = None
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps"  : 20,
    }
    
    def __init__(self,
                 ego_obs=True,
                 n_food=1,
                 activity_range=1.5 / 2,
                 robot_object_spacing=0.3,
                 catch_range=0.25,
                 n_bins=20,
                 sensor_range=1.5,
                 sensor_span=2 * math.pi,
                 coef_inner_rew=0.,
                 coef_main_rew=100.,  # Yoshida et al. 2021 setting
                 coef_ctrl_cost=0.001,  # Yoshida et al. 2021 setting
                 coef_head_angle=0.005,  # Yoshida et al. 2021 setting
                 coef_position_cost=100.,
                 dying_cost=-10,
                 max_episode_steps=60_000,
                 show_sensor_range=False,
                 reward_setting="homeostatic_shaped",
                 reward_bias=None,
                 internal_reset="random",
                 energy_random_range=(0.6, 1.0),
                 temperature_random_range=(40. - 2, 40. + 2.),  # originally +-2
                 visualize_temp=False,
                 show_move_line=False,
                 vision=False,
                 width=32,
                 height=32,
                 random_position=False,
                 domain_randomization=False,
                 no_wall=False,
                 realmode=False,
                 position_homeostasis=False,
                 no_position_obs=False,
                 average_temperature=False,
                 energy_only=False,
                 metabolic_update=0.0001,  # measured by robot1, random policy
                 *args, **kwargs):
        """

        :param int n_food:  Number of greens in each episode
        :param float activity_range: he span for generating objects (x, y in [-range, range])
        :param float robot_object_spacing: Number of objects in each episode
        :param float catch_range: Minimum distance range to catch an food
        :param float shade_range: Maximum distance range to be inside the shade
        :param int n_bins: Number of objects in each episode
        :param float sensor_range: Maximum sensor range (how far it can go)
        :param float sensor_span: Maximum sensor span (how wide it can span), in radians
        :param coef_inner_rew:
        :param coef_main_rew:
        :param coef_cost:
        :param coef_head_angle:
        :param dying_cost:
        :param max_episode_steps:
        :param show_sensor_range: Show range sensor. Default OFF
        :param reward_setting: Setting of the reward definitions. "homeostatic", "homeostatic_shaped", "one", "homeostatic_biased" or "greedy". "homeostatic_shaped" is default. "greedy is not a homeostatic setting"
        :param reward_bias: biasing reward with constant. new_reward = reward + reward_bias
        :param internal_reset: resetting rule of the internal nutrient state. "full" or "random".
        :param energy_random_range: if reset condition is "random", use this region for initialize energy variable
        :param temperature_random_range: if reset condition is "random", use this region for initialize temperature variable (in Celsius degree)
        :param visualize_temp: whether visualize the temperature on the body or not
        :param show_move_line: render the movement of the agent in the environment
        :param vision: enable vision outputs
        :param width: vision width
        :param height: vision height
        :param random_position: set random position at environment reset
        :param args:
        :param kwargs:
        """
        self.n_food = n_food
        self.activity_range = activity_range
        self.robot_object_spacing = robot_object_spacing
        self.catch_range = catch_range
        self.n_bins = n_bins
        self.sensor_range = sensor_range
        self.sensor_span = sensor_span
        self.coef_inner_rew = coef_inner_rew
        self.coef_main_rew = coef_main_rew
        self.coef_ctrl_cost = coef_ctrl_cost
        self.coef_head_angle = coef_head_angle
        self.coef_position_cost = coef_position_cost
        self.dying_cost = dying_cost
        self._max_episode_steps = max_episode_steps
        self.show_sensor_range = show_sensor_range
        self.reward_setting = reward_setting
        self.reward_bias = reward_bias if reward_bias else 0.
        self.internal_reset = internal_reset
        self.energy_random_range = energy_random_range
        self.temperature_random_range = temperature_random_range
        self.show_move_line = show_move_line
        self.visualize_temp = visualize_temp
        self.random_position_at_reset = random_position
        self.domain_randomization = domain_randomization
        self.no_wall = no_wall
        self.realmode = realmode
        self.position_homeostasis = position_homeostasis
        self.no_position_obs = no_position_obs
        self.average_temperature = average_temperature
        self.energy_only = energy_only
        
        if realmode:
            self.n_food = 1
        
        self.objects = []
        
        self.width = width
        self.height = height
        self.vision = vision
        
        # Internal state
        self.full_battery = 1.0
        
        if self.internal_reset in {"full", "random"}:
            self.internal_state = {
                InteroClass.ENERGY     : self.full_battery,
                InteroClass.TEMPERATURE: np.zeros(8) if self.average_temperature else np.zeros(1),
            }
        else:
            raise ValueError
        
        self.prev_interoception = self.get_interoception()
        self.metabolic_update = metabolic_update
        self.energy_lower_bound = 0.0
        
        # Thermal Dynamics Parameters
        
        # Temperature configuration in Celsius degree
        self.target_temperature = 40.
        # Viability range of the robot
        self.temp_limit_max = 60.
        self.temp_limit_min = 20.

        temp_init = np.array([self.target_temperature] * 8, dtype=np.float32)
        self.thermal_model = ThermalModuleParametric3(temp_init=temp_init)  # note: all methods of this model use temperature in Celsius degree

        if self.average_temperature:
            scaled_target_temp = self.scale_temperature(self.thermal_model.get_average_temp())
        else:
            scaled_target_temp = self.scale_temperature(self.thermal_model.get_temp_now())

        if self.energy_only:
            self._target_internal_state = np.array([0.8,])  # [Energy]
        else:
            self._target_internal_state = np.concatenate([[0.8, ], scaled_target_temp])  # [Energy, Temperature]
            
        utils.EzPickle.__init__(**locals())
        
        # for openai baseline
        self.reward_range = (-float('inf'), float('inf'))
        model_cls = self.__class__.MODEL_CLASS
        if model_cls is None:
            raise Exception("MODEL_CLASS unspecified!")
        
        import pathlib
        p = pathlib.Path(inspect.getfile(self.__class__))
        self.MODEL_DIR = os.path.join(p.parent, "models", model_cls.FILE)
        
        # build mujoco
        self.wrapped_env = model_cls(
            self.MODEL_DIR,
            vision=vision,
            width=width,
            height=height,
            no_wall=no_wall,
            activity_range=activity_range,
            domain_randomization=domain_randomization,
            **kwargs
        )
        
        # optimization, caching obs spaces
        self.obs_size = self.wrapped_env.observation_space.shape[0] + self.n_bins
        if self.energy_only:
            self.obs_size += 1
        elif self.average_temperature:
            self.obs_size += 2
        else:
            self.obs_size += 9

        if not self.no_position_obs:
            self.obs_size += 2
            
        ub = BIG * np.ones((self.obs_size,), dtype=np.float32)
        self.obs_space = spaces.Box(ub * -1, ub)
        ub = BIG * np.ones(self.wrapped_env.observation_space.shape, dtype=np.float32)
        self.robot_obs_space = spaces.Box(ub * -1, ub)
        
        # Augment the action space
        ub = np.ones(len(self.wrapped_env.action_space.high), dtype=np.float32)
        self.act_space = spaces.Box(ub * -1, ub)
        
        self.max_episode_length = self._max_episode_steps
        
        self._step = 0
        
        self.num_food_eaten = 0
        
        self.leaf_height = 1
        self.leaf_height_var = 2
        
        # visualization
        self.agent_positions = deque(maxlen=300)
        
    def build_model(self):
        tree = ET.parse(self.MODEL_DIR)
        self.wrapped_env.build_model(tree)
    
    @property
    def dim_intero(self):
        return np.prod(self._target_internal_state.shape)
    
    def reset_internal_state(self, temp_reset):
        # energy
        if self.internal_reset == "full":
            energy = self.full_battery
        elif self.internal_reset == "random":
            energy = np.random.uniform(self.energy_random_range[0], self.energy_random_range[1])
        else:
            raise ValueError
        
        # tempearture
        if temp_reset is not None:
            assert temp_reset.size == 8
            temp = temp_reset.copy()
        else:
            if self.internal_reset == "full":
                temp = np.array([self.target_temperature] * 8, dtype=np.float32)
            
            elif self.internal_reset == "random":
                temp = self.wrapped_env.np_random.uniform(size=8,
                                                          low=self.temperature_random_range[0],
                                                          high=self.temperature_random_range[1])
            else:
                raise ValueError
    
        self.thermal_model.reset(temp_init=temp)
        if self.average_temperature:
            temp_init = self.thermal_model.get_average_temp()
        else:
            temp_init = self.thermal_model.get_temp_now()
        
        self.internal_state = {
            InteroClass.ENERGY     : energy,
            InteroClass.TEMPERATURE: self.scale_temperature(temp_init),
        }
    
    def set_random_position(self):
        L = self.activity_range - self.activity_range * 0.1
        self.wrapped_env.data.qpos[:2] = self.np_random.uniform(-L, L, size=2)
        
        random_angle = self.np_random.uniform(0, 2 * np.pi)
        q = eulertoq(np.array([0, 0, random_angle]))
        if self.wrapped_env.IS_WALKER:
            self.wrapped_env.data.qpos[3:3 + 4] = q
        else:
            self.wrapped_env.data.qpos[2:2 + 2] = q[:2]
    
    def reset(self, seed=None, return_info=True, n_food=None, options=None):
        self._step = 0
        
        if n_food is not None and not self.realmode:
            self.n_food = n_food
        
        self.num_food_eaten = 0
        
        # set random position
        if self.random_position_at_reset:
            self.set_random_position()

        self.wrapped_env.reset(seed=seed)
        
        initial_temp = None
        if options is not None:
            initial_temp = options.get("initial_temp")
        self.reset_internal_state(initial_temp)
        
        self.prev_interoception = self.get_interoception()
        self.agent_positions.clear()
        
        com = self.wrapped_env.get_body_com("torso")
        robo_x, robo_y = com[:2]
        
        self.objects = []
        existing = set()
        while len(self.objects) < self.n_food:
            x, y, z = self.new_position(self.realmode)
            
            # regenerate, since it is too close to the robot's initial position
            if (x / 10. - robo_x) ** 2 + (y / 10. - robo_y) ** 2 < self.robot_object_spacing ** 2:
                continue

            if (x, y) in existing:
                continue

            typ = ObjectClass.FOOD
            self.objects.append((x, y, z, typ))
            existing.add((x, y))
        
        info = {"interoception": self.get_interoception()}
        
        return (self.get_current_obs(), info) if return_info else self.get_current_obs()
    
    def new_position(self, realmode: bool):
        if realmode:
            x = self.wrapped_env.np_random.choice([int(-10 * 0.5), int(10 * 0.5)])
            y = self.wrapped_env.np_random.choice([int(-10 * 0.5), int(10 * 0.5)])
        else:
            x = self.wrapped_env.np_random.integers(-10 * self.activity_range / 2, 10 * self.activity_range / 2) * 2
            y = self.wrapped_env.np_random.integers(-10 * self.activity_range / 2, 10 * self.activity_range / 2) * 2

        z = 0.1
        return x, y, z
    
    def generate_new_object(self, robo_x: float, robo_y: float, type_gen: ObjectClass):
        existing = set()
        for object in self.objects:
            existing.add((object[0], object[1]))
        
        while True:
            x, y, z = self.new_position(self.realmode)
            
            if (x, y) in existing:
                continue
            
            if (x / 10. - robo_x) ** 2 + (y / 10. - robo_y) ** 2 < self.robot_object_spacing ** 2:
                continue
                
            return (x, y, z, type_gen)
    
    def step(self, action: np.ndarray):
        action = np.clip(action, a_min=-1, a_max=1)
        
        motor_action = action
        evaporative_action = 0
        
        self.prev_interoception = self.get_interoception()
        self.prev_robot_xy = self.wrapped_env.get_body_com("torso")[:2].copy()
        _, inner_rew, terminated, truncated, info = self.wrapped_env.step(motor_action)
        truncated = False
        
        info['inner_rew'] = inner_rew
        # com = self.wrapped_env.get_body_com("front_left_ankle")
        # xl, yl = com[:2]
        # com = self.wrapped_env.get_body_com("front_right_ankle")
        # xr, yr = com[:2]
        com = self.wrapped_env.get_body_com("torso")
        x, y = com[:2]
        self.agent_positions.append(np.array(com, np.float32))
        info['com'] = com
        
        #  Default Metabolic update
        self.internal_state[InteroClass.ENERGY] -= self.metabolic_update
        
        # Food-Eating
        new_objs = []
        self.num_food_eaten = 0
        for obj in self.objects:
            ox, oy, oz, typ = obj
            # object within zone!
            if typ is ObjectClass.FOOD:
                if (ox / 10. - x) ** 2 + (oy / 10. - y) ** 2 < self.catch_range ** 2:
                    self.num_food_eaten += 1
                    new_objs.append(self.generate_new_object(robo_x=x, robo_y=y, type_gen=typ))
                    
                    # self.thermal_model.reset(self.target_temperature)
                    self.internal_state[InteroClass.ENERGY] = self.full_battery
                else:
                    new_objs.append(obj)
        
        self.objects = new_objs
        
        # Update thermal step
        self.thermal_model.step(motor_action=motor_action,
                                joint_position=self.wrapped_env.get_joint_position(),
                                dt=self.dt,
                                mode="RK4")

        if self.average_temperature:
            temp_now = self.thermal_model.get_average_temp()
        else:
            temp_now = self.thermal_model.get_temp_now()

        self.internal_state[InteroClass.TEMPERATURE] = self.scale_temperature(temp_now)
        
        info["interoception"] = self.get_interoception()
        
        terminated = False
        if self.get_interoception()[0] < self.energy_lower_bound:  # done if energy level is sufficiently low
            terminated = True
        
        # turn off thermal terminal if energy_only mode is enabled
        if not self.energy_only and np.sum(self.get_interoception()[1:] ** 2 >= 1.0) != 0:
            terminated = True
        
        info["food_eaten"] = (self.num_food_eaten,)
        
        self._step += 1
        terminated = terminated or self._step >= self._max_episode_steps
        
        reward, info_rew = self.get_reward(reward_setting=self.reward_setting,
                                           action=action,
                                           done=terminated)
        
        info.update(info_rew)
        
        return self.get_current_obs(), reward, terminated, truncated, info
    
    def get_reward(self, reward_setting, action, done):

        info = {"reward_module": None, "position_cost": None}
        
        def drive(intero, target, weight=None):
            drive_module = -1 * (intero - target) ** 2
            if weight is not None:
                drive_module = drive_module * weight  # weighted sum

            d_ = drive_module.sum()
            return d_, drive_module

        # Motor Cost
        lb, ub = self.action_space.low, self.action_space.high
        scaling = (ub - lb) * 0.5
        ctrl_cost = -.5 * np.square((action - FLAT_POSTURE_ACTION) / scaling).sum()
        
        # Local Posture Cost
        if self.wrapped_env.IS_WALKER:
            euler = qtoeuler(self.wrapped_env.data.qpos[self.__class__.ORI_IND:self.__class__.ORI_IND + 4])
            euler_stand = qtoeuler([1.0, 0.0, 0.0, 0.0])  # quaternion of standing state
            head_angle_cost = -np.square(euler[:2] - euler_stand[:2]).sum()  # Drop yaw
        else:
            head_angle_cost = 0.

        # Global Positional Cost
        if self.position_homeostasis:
            d, dm = drive(self.wrapped_env.get_body_com("torso")[:2], np.zeros(2))
            d_prev, dm_prev = drive(self.prev_robot_xy, np.zeros(2))
            position_cost = d - d_prev
        else:
            body_xy = self.wrapped_env.get_body_com("torso")[:2]
            position_cost = - (body_xy[0] ** 2 + body_xy[1] ** 2)
            
        info["position_cost"] = position_cost

        total_cost = self.coef_ctrl_cost * ctrl_cost + self.coef_head_angle * head_angle_cost + self.coef_position_cost * position_cost
        
        # Main Reward
        if reward_setting == "homeostatic":
            d, dm = drive(self.prev_interoception, self._target_internal_state, DRIVE_WEIGHTS[:self.dim_intero])
            main_reward = d
            info["reward_module"] = np.concatenate([self.coef_main_rew * dm, [total_cost]])
        
        elif reward_setting == "homeostatic_shaped":
            d, dm = drive(self.get_interoception(), self._target_internal_state, DRIVE_WEIGHTS[:self.dim_intero])
            d_prev, dm_prev = drive(self.prev_interoception, self._target_internal_state, DRIVE_WEIGHTS[:self.dim_intero])
            main_reward = d - d_prev
            info["reward_module"] = np.concatenate([self.coef_main_rew * (dm - dm_prev), [total_cost]])
        
        elif reward_setting == "one":
            # From continual-Cartpole setting from the lecture of Doina Precup (EEML 2021).
            if done:
                main_reward = -1.
            else:
                main_reward = 0.
        
        elif reward_setting == "homeostatic_biased":
            d, dm = drive(self.prev_interoception, self._target_internal_state, DRIVE_WEIGHTS[:self.dim_intero])
            main_reward = d + self.reward_bias
            info["reward_module"] = np.concatenate([self.coef_main_rew * dm, [total_cost]])
        
        else:
            raise ValueError
        
        reward = self.coef_main_rew * main_reward + total_cost
        info["error"] = -drive(self.get_interoception(), self._target_internal_state, DRIVE_WEIGHTS[:self.dim_intero])[0]
        
        return reward, info
    
    def get_readings(self):  # equivalent to get_current_maze_obs in maze_env.py
        # compute sensor readings
        # first, obtain current orientation
        food_readings = np.zeros(self.n_bins)
        robot_x, robot_y = self.wrapped_env.get_body_com("torso")[:2]
        # sort objects by distance to the robot, so that farther objects'
        # signals will be occluded by the closer ones'
        
        sorted_objects = sorted(
            self.objects, key=lambda o:
            (o[0] / 10. - robot_x) ** 2 + (o[1] / 10. - robot_y) ** 2)[::-1]
        
        # print(f"robo (x,y): {robot_x}, {robot_y}")
        # print(f"food (x,y): {sorted_objects[0][0]}, {sorted_objects[0][1]}")

        # fill the readings
        bin_res = self.sensor_span / self.n_bins
        
        ori = self.get_ori()
        
        for ox, oy, oz, typ in sorted_objects:
            # compute distance between object and robot
            dist = ((oy / 10. - robot_y) ** 2 + (ox / 10. - robot_x) ** 2) ** 0.5
            # print("---- dist : ", dist)

            # only include readings for objects within range
            if dist > self.sensor_range:
                continue
            
            angle = math.atan2(oy / 10. - robot_y, ox / 10. - robot_x) - ori
            # print(f"ori: {ori}, angle: {angle}")
            
            if math.isnan(angle):
                import ipdb;
                ipdb.set_trace()
            
            angle = angle % (2 * math.pi)
            
            if angle > math.pi:
                angle = angle - 2 * math.pi
            
            if angle < -math.pi:
                angle = angle + 2 * math.pi
            
            # outside of sensor span - skip this
            half_span = self.sensor_span * 0.5
            
            if abs(angle) > half_span:
                continue
            
            bin_number = int((angle + half_span) / bin_res)
            if bin_number == len(food_readings):
                bin_number -= len(food_readings)

            intensity = 1.0 - dist / self.sensor_range
            
            if typ is ObjectClass.FOOD:
                food_readings[bin_number] = intensity
        
        return food_readings
    
    def get_interoception(self):
        energy = self.internal_state[InteroClass.ENERGY]
        motor_temp = self.internal_state[InteroClass.TEMPERATURE]
        if self.energy_only:
            return np.array([energy], dtype=np.float32)
        else:
            return np.concatenate([[energy], motor_temp], dtype=np.float32)
    
    def get_current_robot_obs(self):
        return self.wrapped_env.get_current_obs()
    
    def get_current_obs(self):
        # return sensor data along with data about itself
        self_obs = self.wrapped_env.get_current_obs()
        food_readings = self.get_readings()
        body_xy = self.wrapped_env.get_body_com("torso")[:2]
        interoception = self.get_interoception()
        
        if self.no_position_obs:
            obs = np.concatenate([self_obs, food_readings, interoception], dtype=np.float32)
        else:
            obs = np.concatenate([self_obs, food_readings, body_xy, interoception], dtype=np.float32)

        return obs
    
    @property
    def multi_modal_dims(self):
        proprioception_dim = self.robot_obs_space.shape[0]
        food_readings = self.n_bins
        exteroception_dim = food_readings
        if not self.no_position_obs:
            exteroception_dim += 2
        
        # (proprioception, exteroception, interoception)
        interoception_dim = len(self.get_interoception())
        return tuple([proprioception_dim, exteroception_dim, interoception_dim])
    
    @property
    def observation_space(self):
        return self.obs_space
    
    # space of only the robot observations (they go first in the get current obs)
    @property
    def robot_observation_space(self):
        return self.robot_obs_space
    
    @property
    def action_space(self):
        return self.act_space
    
    @property
    def dt(self):
        return self.wrapped_env.dt
    
    def close(self):
        if self.wrapped_env.mujoco_renderer is not None:
            self.wrapped_env.mujoco_renderer.close()
    
    def scale_temperature(self, temperature):
        """
        Scale the temperature in Celsius degree into the range [-1, 1]
        :param temperature: absolute temperature
        :return:
        """
        out = 2 * (temperature - self.temp_limit_min) / (self.temp_limit_max - self.temp_limit_min) - 1
        return out
    
    def decode_temperature(self, scaled_temperature):
        """
        decode the scaled temperature [-1, 1] into temperature in Celsius degree
        :param scaled_temperature: temperature in scale of [-1, +1]
        :return:
        """
        out = 0.5 * (scaled_temperature + 1) * (self.temp_limit_max - self.temp_limit_min) + self.temp_limit_min
        return out
    
    def get_ori(self):
        """
        First it tries to use a get_ori from the wrapped env. If not successfull, falls
        back to the default based on the ORI_IND specified in Maze (not accurate for quaternions)
        """
        obj = self.wrapped_env
        while not hasattr(obj, 'get_ori') and hasattr(obj, 'wrapped_env'):
            obj = obj.wrapped_env
        try:
            return obj.get_ori()
        except (NotImplementedError, AttributeError) as e:
            pass
        return self.wrapped_env.data.qpos[self.__class__.ORI_IND]
    
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
            camera_id=1,
            camera_name=None
    ):
        
        if mode == "rgbd_array":
            viewers = [self.wrapped_env.mujoco_renderer._get_viewer(render_mode="rgb_array")]
            viewers.append(self.wrapped_env.mujoco_renderer._get_viewer(render_mode="depth_array"))
        else:
            viewers = [self.wrapped_env.mujoco_renderer._get_viewer(render_mode=mode)]
        
        # Show Sensor Range
        if self.show_sensor_range:
            
            reading = self.get_readings()
            # print(reading)
            robot_x, robot_y = self.wrapped_env.get_body_com("torso")[:2]
            ori = self.get_ori()
            for n in range(self.n_bins):
                theta = ori + self.sensor_span * ((n + 1) / self.n_bins - 0.5)
                
                if reading[n] < 1e-3:
                    intensity = 0
                else:
                    intensity = self.sensor_range * (1 - reading[n])
            
                ox = robot_x + intensity* math.cos(theta)
                oy = robot_y + intensity * math.sin(theta)
                for v in viewers:
                    v.add_marker(
                        pos=np.array([ox, oy, 0.3]),
                        label=" ",
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=(0.03, 0.03, 0.03),
                        rgba=(n / self.n_bins, 0, 0, 0.8)
                    )
                    
            # showing capture range
            for v in viewers:
                v.add_marker(
                    pos=np.array([robot_x, robot_y, 0.1]),
                    label=" ",
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=(self.catch_range, self.catch_range, 0.0),
                    rgba=(1, 1, 0, 0.3)
                )
        
        # show movement of the agent
        if self.show_move_line:
            for pos in self.agent_positions:
                for v in viewers:
                    v.add_marker(pos=pos,
                                 label=" ",
                                 type=mujoco.mjtGeom.mjGEOM_SPHERE,
                                 size=(0.05, 0.05, 0.05),
                                 rgba=(1, 0, 0, 0.3),
                                 emission=1)
        
        # Show body core temperature
        if self.visualize_temp:
            raw_temp = np.clip(2 * self.get_interoception()[1], -1, 1)
            col = np.sign(self.get_interoception()[1])
            robot_x, robot_y, robo_z = self.wrapped_env.get_body_com("torso")[:3]
            for v in viewers:
                v.add_marker(
                    pos=np.array([robot_x, robot_y, robo_z]),
                    label=" ",
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=(.2, .2, .2),
                    rgba=(col, 0, 1 - col, 0.8 * np.abs(raw_temp))
                )
        
        # Show food
        for obj in self.objects:
            ox, oy, oz, typ = obj
            if typ is ObjectClass.FOOD:
                for v in viewers:
                    v.add_marker(
                        pos=np.array([ox / 10., oy / 10., oz]),
                        label=" ",
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=(0.1),
                        rgba=(1, 0, 0, 1)
                    )
                
        if mode == "rgbd_array":
            im = self.wrapped_env.mujoco_renderer.render(
                "rgb_array",
                camera_id,
                camera_name, )
            im_d = self.wrapped_env.mujoco_renderer.render(
                "depth_array",
                camera_id,
                camera_name, )[:, :, np.newaxis]
            
            im = np.dstack((im, im_d))
        else:
            im = self.wrapped_env.mujoco_renderer.render(
                mode,
                camera_id,
                camera_name,
            )
        
        # delete unnecessary markers: https://github.com/openai/mujoco-py/issues/423#issuecomment-513846867
        for v in viewers:
            del v._markers[:]
        
        return im
