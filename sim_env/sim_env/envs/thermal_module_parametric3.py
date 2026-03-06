"""
Parametric thermal model of the robot. Parameters were fitted by using CMA-ES

Absolute temperature is used only in the module, and all methods use Celsius degree.
"""
import json
import os
import sys

import numpy as np


def to_absolute(temp_celsius: np.ndarray):
    return temp_celsius + 273.

def to_celsius(temp_absolute: np.ndarray):
    return temp_absolute - 273.


class ThermalModuleParametric3:
    
    def __init__(self, temp_init):
        """

        :param temp_init: Initial body temperature in Celsius degree
        """
        assert temp_init.size == 8
        self._T = to_absolute(temp_init)
        self._initial_T = to_absolute(temp_init)
        self._T_smooth = temp_init

        param_files = [
            "best_hip1_ind_param.json",
            "best_ankle1_ind_param.json",
            "best_hip2_ind_param.json",
            "best_ankle2_ind_param.json",
            "best_hip3_ind_param.json",
            "best_ankle3_ind_param.json",
            "best_hip4_ind_param.json",
            "best_ankle4_ind_param.json",
        ]
        param_files = [os.path.dirname(__file__) + "/thermal_params/20230823/" + f for f in param_files]
        
        self._CAPACITY = []
        self._RESISTANCE = []
        self._T_BG = [to_absolute(25)] * 8
        self._HEAT_GEN_COEF = []
        for f in param_files:
            with open(f, "r") as data:
                param = json.load(data)
                self._CAPACITY.append(param[0])
                self._RESISTANCE.append(param[1])
                self._HEAT_GEN_COEF.append(param[2:])
        
        self._CAPACITY = np.array(self._CAPACITY, dtype=np.float32)
        self._RESISTANCE = np.array(self._RESISTANCE, dtype=np.float32)
        self._T_BG = np.array(self._T_BG, dtype=np.float32)  # background temperature
        self._HEAT_GEN_COEF = np.vstack(self._HEAT_GEN_COEF)
        
    def reset(self, temp_init=None):
        """

        :param temp_init: Initial body temperature in Celsius degree
        :return:
        """
        if temp_init is not None:
            assert temp_init.size == 8
            self._initial_T = to_absolute(temp_init)

        self._T = self._initial_T.copy()
        self._T_smooth = np.asarray(temp_init, dtype=np.int64)
        
    def calq_Q_in(self, joint, a, b, c):
        epsilon = np.zeros_like(joint)
        epsilon[::2] = np.abs(joint[::2])  # hip
        epsilon[1::2] = joint[1::2]  # ankle
        
        # Avoid numerical overflow
        x = a * (epsilon - b)
        index_overflow = np.abs(x) > 100
        x[index_overflow] = np.sign(x[index_overflow]) * 100
        
        return c / (1 + np.exp(-x))

    def _delta_Q(self, T_now, motor_action, joint_position):
        """
        Assuming all actions sould be scaled into [-1, +1]
        """
        
        q_in = self.calq_Q_in(
            joint_position,
            a=self._HEAT_GEN_COEF[:, 0],
            b=self._HEAT_GEN_COEF[:, 1],
            c=self._HEAT_GEN_COEF[:, 2],
        )

        q_out = (T_now - self._T_BG) / self._RESISTANCE
        
        dQ = q_in - q_out
        
        return dQ

    def _grad_T(self, T_now, motor_action, joint_position):
        return self._delta_Q(T_now, motor_action, joint_position) / self._CAPACITY

    def step(self, motor_action, joint_position, dt, mode="RK4"):
        """
        One-step progress of the thermal model. return the latest body temperature in Celsius degree
        """
        assert self._T is not None and self._T_smooth is not None

        # Eular
        if mode == "Euler":
            T_now = self._T
            dT = self._grad_T(T_now, motor_action, joint_position)
            dT *= dt
        elif mode == "RK4":
            # RK4 based on https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#The_Runge%E2%80%93Kutta_method
            T_now = self._T
            k1 = self._grad_T(T_now, motor_action,joint_position)
            k2 = self._grad_T(T_now + dt * k1 / 2., motor_action, joint_position)
            k3 = self._grad_T(T_now + dt * k2 / 2., motor_action, joint_position)
            k4 = self._grad_T(T_now + dt * k3, motor_action, joint_position)
            dT = (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6.
        else:
            raise ValueError("mode error")

        self._T += dT
        
        # Smoothing
        # self._T_smooth = self._T_smooth + 0.01 * (np.asarray(self._T, dtype=np.int64) - self._T_smooth)
        self._T_smooth = self._T_smooth + 0.01 * (np.asarray(to_celsius(self._T), dtype=np.int64) - self._T_smooth)

        return to_celsius(self._T)

    def get_temp_now(self):
        # return to_celsius(self._T_smooth)
        return self._T_smooth.copy()

    def get_raw_temp(self):
        return to_celsius(self._T)
    
    def get_average_temp(self):
        return self.get_temp_now().mean(keepdims=True)


if __name__ == '__main__':
    model = ThermalModuleParametric3(temp_init=np.zeros(8) + 38)
    # q = model._delta_Q(T_now=np.zeros(8) + 38, motor_action=np.ones(8), joint_position=np.zeros(8) + 0.5)
    # print(q)
    
    print(model.get_temp_now())
    print(model.get_raw_temp())
    
    model.reset(temp_init=np.zeros(8) + 40)

    print(model.get_temp_now())
    print(model.get_raw_temp())
