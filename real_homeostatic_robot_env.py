import collections
import math
from typing import Optional

import numpy as np
import gymnasium
import pygame

from gymnasium import spaces
from realant_env import RealAntEnv
from utils import qtoeuler, get_motor_temperature

M2PIX = 200
field_size = 1.5
window = pygame.display.set_mode((int(field_size * M2PIX), int(field_size * M2PIX)))
food_col = (200, 0, 0)
food_pos_suggested_col = (200, 150, 150)
robot_col = (255, 255, 255)
radius = int(0.07 * M2PIX)
robot_size = int(0.1 * M2PIX)
pos_offset = (field_size / 2) * M2PIX

FOOD_POSITIONS = [
    np.array([-0.3, -0.3], dtype=np.float32),
    np.array([-0.3, 0.3], dtype=np.float32),
    np.array([0.3, -0.3], dtype=np.float32),
    np.array([0.3, 0.3], dtype=np.float32),
]


class RealHomeostaticRobotEnv(gymnasium.Env):
    def __init__(self,
                 obs_stack=3,
                 action_as_obs=True,  # set previous action as a component of obs
                 is_v2=True,  # thermal model is newer than v2
                 no_position_obs=True,
                 energy_only=False,
                 avg_temp=False,
                 no_food=False,
                 ):

        # Fixme: set appropriate params in sim
        self.n_bins = 20
        self.sensor_range = 1.5
        self.sensor_span = 2 * math.pi
        self.is_v2 = is_v2
        self.no_position_obs = no_position_obs
        self.energy_only = energy_only
        self.avg_temp = avg_temp
        self.no_food = no_food

        self.wrapped_env = RealAntEnv(
            obs_stack=obs_stack,
            action_as_obs=action_as_obs,  # set previous action as a component of obs
            joint_only=True,
        )

        self._smoothed_temperature = None

        self.target_temperature = np.array([40.] * 8)
        # Viability range of the robot
        self.temp_limit_max = 60.
        self.temp_limit_min = 20.

        self.action_space = self.wrapped_env.action_space

        n_prop = 8 * obs_stack + 8 * int(action_as_obs) + 2  # joint + prev action + xy
        n_extero = self.n_bins  # sensor
        if energy_only:
            n_intero = 1
        elif avg_temp:
            n_intero = 2
        else:
            n_intero = 1 + 8 if self.is_v2 else 2  # energy + joint temp
        self.obs_shape = n_prop + n_extero + n_intero
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_shape,), dtype=np.float64)

        self.intero_monitor = collections.deque(maxlen=200)
        self.charge_initial = None

        self.food_position_suggested = None
        self.food_move_thresh = 0.05  # 5cm
        self.prev_food_pos = None
        self.robot_object_spacing = 0.3  # 30cm

    def update_smoothed_temp(self, obs_dict):
        # re-ordering from servo order to simulation order
        temps_now = get_motor_temperature(obs_dict)

        if self._smoothed_temperature is None:
            self._smoothed_temperature = temps_now
        else:
            # Smoothing of the temperature, same parameter with the thermal model
            self._smoothed_temperature = self._smoothed_temperature + 0.01 * (temps_now - self._smoothed_temperature)

    def reset(self,
              seed: Optional[int] = None,
              options: Optional[dict] = None,
              ):
        _, info_base = self.wrapped_env.reset(seed, options)

        self._smoothed_temperature = None

        self.update_smoothed_temp(self.wrapped_env.obs_dict_now)

        self.suggest_new_food_pos(self.wrapped_env.obs_dict_now)

        _, food_pos_now = self.detect_food_move(self.wrapped_env.obs_dict_now)
        self.prev_food_pos = None
        is_food_captured = False

        obs, info_obs = self.get_current_obs(self.wrapped_env.obs_dict_now)

        info = {"is_food_captured": is_food_captured, "food_pos": food_pos_now}
        info.update(self.wrapped_env.obs_dict_now)
        info.update(info_base)
        info.update(info_obs)

        return obs, info

    def step(self, action):
        _, _, _, _, info_base = self.wrapped_env.step(action)

        self.update_smoothed_temp(self.wrapped_env.obs_dict_now)

        is_food_captured, food_pos_now = self.detect_food_move(self.wrapped_env.obs_dict_now)

        obs, info_obs = self.get_current_obs(self.wrapped_env.obs_dict_now)

        reward = 0

        terminal = False
        truncated = False

        info = {"is_food_captured": is_food_captured, "food_pos": food_pos_now}
        info.update(self.wrapped_env.obs_dict_now)
        info.update(info_base)
        info.update(info_obs)

        return obs, reward, terminal, truncated, info

    def suggest_new_food_pos(self, obs_dict):
        robot_pos = np.array([obs_dict["ANT"][0][2], obs_dict["ANT"][0][0]])

        new_pos = robot_pos.copy()

        while np.linalg.norm(robot_pos - new_pos) < self.robot_object_spacing:
            position_id = self.wrapped_env.np_random.integers(0, len(FOOD_POSITIONS))
            new_pos = FOOD_POSITIONS[position_id].copy()

        self.prev_food_pos = None
        self.food_position_suggested = new_pos

    def scale_temperature(self, temperature):
        """
        Scale the temperature in Celsius degree into the range [-1, 1]
        :param temperature: absolute temperature
        :return:
        """
        out = 2 * (temperature - self.temp_limit_min) / (self.temp_limit_max - self.temp_limit_min) - 1
        return out

    def detect_food_move(self, obs_dict):

        detection = False
        food_pos = np.array([obs_dict["FOOD"][0][2], obs_dict["FOOD"][0][0]], dtype=np.float32)

        if self.prev_food_pos is None:
            self.prev_food_pos = food_pos.copy()

        if np.linalg.norm(food_pos - self.prev_food_pos) > self.food_move_thresh:
            detection = True
            self.prev_food_pos = None

        return detection, food_pos

    def get_current_obs(self, obs_dict):
        # return sensor data along with data about itself
        self_obs = self.wrapped_env.get_current_obs(obs_dict)

        food_readings = self.get_readings(obs_dict)

        body_xy = self.get_body_xyz(obs_dict)[:2]

        interoception, charge, charge_norm, temps, normalized_smooth_temp = self.get_interoception(obs_dict)

        info = {
            "charge_normal": charge_norm,
            "temp_normal": normalized_smooth_temp,
            "charge": charge,
            "temp": temps,
        }

        if self.no_position_obs:
            return np.concatenate([self_obs, food_readings, interoception], dtype=np.float32), info
        else:
            return np.concatenate([self_obs, food_readings, body_xy, interoception], dtype=np.float32), info

    def get_readings(self, obs_dict):
        # compute sensor readings
        # first, obtain current orientation
        food_readings = np.zeros(self.n_bins)
        robot_x, robot_y = obs_dict["ANT"][0][2], obs_dict["ANT"][0][0]
        # sort objects by distance to the robot, so that farther objects'
        # signals will be occluded by the closer ones'

        sorted_objects = [(obs_dict["FOOD"][0][2], obs_dict["FOOD"][0][0]), ]
        # optitrack --> playroom: xyz --> yzx

        # print(f"robo (x,y): {robot_x}, {robot_y}")
        # print(f"food (x,y): {sorted_objects[0][0]}, {sorted_objects[0][1]}")

        # fill the readings
        bin_res = self.sensor_span / self.n_bins

        ori = self.get_ori(obs_dict)

        for ox, oy in sorted_objects:
            # compute distance between object and robot
            dist = ((oy - robot_y) ** 2 + (ox - robot_x) ** 2) ** 0.5
            # print("---- dist : ", dist)

            # only include readings for objects within range
            if dist > self.sensor_range:
                continue

            angle = math.atan2(oy - robot_y, ox - robot_x) - ori
            # print(f"ori: {np.rad2deg(ori)}, angle: {angle}")

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

            food_readings[bin_number] = intensity

        if self.no_food:
            food_readings *= 0

        return food_readings

    def get_ori(self, obs_dict):
        # print("ant posture: ", obs_dict["ANT"][1])
        body_rpy_ = qtoeuler(obs_dict["ANT"][1])  # pitch, yaw, roll
        yaw = body_rpy_[1]

        if np.abs(body_rpy_[0]) > np.pi * 0.8 and np.abs(body_rpy_[2]) > np.pi * 0.8:
            if yaw > 0:
                yaw = np.pi - yaw
            else:
                yaw = -np.pi - yaw

        return yaw

    def get_body_xyz(self, obs_dict):
        body_xyz = np.array(obs_dict["ANT"][0])
        return body_xyz[2], body_xyz[0], body_xyz[1]  # optitrack to playroom coordination

    def get_normalized_smoothed_temps(self):
        return self.scale_temperature(self._smoothed_temperature), self._smoothed_temperature.copy()

    def get_interoception(self, obs_dict):
        normalized_smoothed_temps = self.scale_temperature(self._smoothed_temperature.copy())

        # voltage = obs_dict["volt"]
        if self.charge_initial is None:
            self.charge_initial = obs_dict["charge"]

        charge = obs_dict["charge"]

        normalized_charge = 1. - (charge - self.charge_initial) / 1000.

        if self.energy_only:
            intero = np.array([normalized_charge])
        elif self.avg_temp:
            intero = np.array([normalized_charge, self.scale_temperature(self._smoothed_temperature.copy().mean())])
        else:
            if self.is_v2:
                intero = np.concatenate([[normalized_charge], normalized_smoothed_temps])
            else:
                intero = np.array([normalized_charge, normalized_smoothed_temps.max()])

        return intero, charge, normalized_charge, self._smoothed_temperature.copy(), normalized_smoothed_temps

    def render(self):
        obs_dict = self.wrapped_env.obs_dict_now

        food_x, food_y = obs_dict["FOOD"][0][2], obs_dict["FOOD"][0][0]

        window.fill((200, 200, 200))

        def real_to_display(x, y):
            return x, -y

        def draw_robot(screen, x, y, ori):

            polygon_vertices = [
                ((x * M2PIX + pos_offset - robot_size * math.cos(-ori - math.pi / 4)),
                 y * M2PIX + pos_offset - robot_size * math.sin(-ori - math.pi / 4)),
                ((x * M2PIX + pos_offset - robot_size * math.cos(-ori + math.pi / 4)),
                 y * M2PIX + pos_offset - robot_size * math.sin(-ori + math.pi / 4)),
                ((x * M2PIX + pos_offset - robot_size * math.cos(-ori + math.pi * 3 / 4)),
                 y * M2PIX + pos_offset - robot_size * math.sin(-ori + math.pi * 3 / 4)),
                ((x * M2PIX + pos_offset - robot_size * math.cos(-ori - math.pi * 3 / 4)),
                 y * M2PIX + pos_offset - robot_size * math.sin(-ori - math.pi * 3 / 4)),
            ]

            pygame.draw.polygon(screen, robot_col, polygon_vertices)

            leg_pos = [
                ((x * M2PIX + pos_offset - 2 * robot_size * math.cos(-ori - math.pi / 4)),
                 y * M2PIX + pos_offset - 2 * robot_size * math.sin(-ori - math.pi / 4)),
                ((x * M2PIX + pos_offset - 2 * robot_size * math.cos(-ori + math.pi / 4)),
                 y * M2PIX + pos_offset - 2 * robot_size * math.sin(-ori + math.pi / 4)),
                ((x * M2PIX + pos_offset - 2 * robot_size * math.cos(-ori + math.pi * 3 / 4)),
                 y * M2PIX + pos_offset - 2 * robot_size * math.sin(-ori + math.pi * 3 / 4)),
                ((x * M2PIX + pos_offset - 2 * robot_size * math.cos(-ori - math.pi * 3 / 4)),
                 y * M2PIX + pos_offset - 2 * robot_size * math.sin(-ori - math.pi * 3 / 4)),
            ]

            for p in leg_pos:
                pygame.draw.line(screen,
                                 robot_col,
                                 (x * M2PIX + pos_offset, y * M2PIX + pos_offset),
                                 p,
                                 8)

            pygame.draw.circle(screen,
                               (0, 0, 0),
                               (x * M2PIX + pos_offset + 0.05 * math.cos(ori) * M2PIX,
                                y * M2PIX + pos_offset - 0.05 * math.sin(ori) * M2PIX),
                               5)

        robot_x, robot_y = real_to_display(obs_dict["ANT"][0][2], obs_dict["ANT"][0][0])
        robot_ori = self.get_ori(obs_dict)
        reading = self.get_readings(obs_dict)
        for n in range(self.n_bins):
            theta = robot_ori + self.sensor_span * ((n + 1) / self.n_bins - 0.5)

            if reading[n] < 1e-3:
                intensity = 0
            else:
                intensity = self.sensor_range * (1 - reading[n])
            pygame.draw.line(
                window,
                (255 * n / self.n_bins, 0, 0),
                (robot_x * M2PIX + pos_offset, robot_y * M2PIX + pos_offset),
                (robot_x * M2PIX + pos_offset + intensity * math.cos(theta) * M2PIX,
                 robot_y * M2PIX + pos_offset - intensity * math.sin(theta) * M2PIX),
                1
            )

        # food position candidates
        for pf in FOOD_POSITIONS:
            pygame.draw.circle(window, (100, 100, 100), np.array(real_to_display(pf[0], pf[1])) * M2PIX + pos_offset,
                               radius)

        # food pos suggested by system
        pygame.draw.circle(window, food_pos_suggested_col, np.array(
            real_to_display(self.food_position_suggested[0], self.food_position_suggested[1])) * M2PIX + pos_offset,
                           radius)

        # food
        if not self.no_food:
            pygame.draw.circle(window, food_col, np.array(real_to_display(food_x, food_y)) * M2PIX + pos_offset, radius)

        # robot
        draw_robot(window, robot_x, robot_y, robot_ori)

        pygame.display.update()
