import enum
from collections import defaultdict

import numpy as np

# Target cycle of the RL loop
TARGET_DURATION = 0.05

# I2C address
DEVICE_ADDR = 0x66

# LTC2947 control address
# https://strawberry-linux.com/pub/j2947f.pdf
# https://strawberry-linux.com/pub/2947fa.pdf (Japanese ver.)
CTRL_ADDR = 0xF0
CURRENT_ADDR = 0x90
VOLT_ADDR = 0xA0
WATT_ADDR = 0x93
CHARGE_ADDR = 0x00
ENERGY_ADDR = 0x06

# Action constants
# hip_range = 256
# hip_offset = 368  # this limits hip from middle to +-45deg
# ankle_range = 224
# ankle_offset = 288
# servo_middle = 512  # ax12a value for servo middle position
# servo_half_range = 512 - 224  # ax12a range from middle to zero degrees
DEGREE_TO_TICK = 1023 / 300
hip_offset = DEGREE_TO_TICK * 150  # center
hip_range = DEGREE_TO_TICK * (80 / 2)
ankle_offset = DEGREE_TO_TICK * (150 - 40)
ankle_range = DEGREE_TO_TICK * (60 / 2)

# Observation constants

KEYS_JOINT = ["s1_angle", "s2_angle", "s3_angle", "s4_angle", "s5_angle", "s6_angle", "s7_angle", "s8_angle"]
KEYS_TEMPERATURE = ["s1_temp", "s2_temp", "s3_temp", "s4_temp", "s5_temp", "s6_temp", "s7_temp", "s8_temp"]

# KEYS_HAT = [
#     "temperature",
#     "humidity",
#     "pressure",
#     "pitch", "roll", "yaw",
#     "accel_x", "accel_y", "accel_z",
#     "gyro_x", "gyro_y", "gyro_z",
# ]

KEYS_BATTERY = [
    "current",
    "volt",
    "watt",
    "charge",
    "energy"
]

KEYS_MOTORS = [
    "ant_time",
    # "server_epoch_ms",
    "s1_angle", "s2_angle", "s3_angle", "s4_angle", "s5_angle", "s6_angle", "s7_angle", "s8_angle",
    #  "s1_sp", "s2_sp", "s3_sp", "s4_sp", "s5_sp", "s6_sp", "s7_sp", "s8_sp",  # set point of servos
    #  "feet1", "feet2", "feet3", "feet4",  # zero all
    "s1_temp", "s2_temp", "s3_temp", "s4_temp", "s5_temp", "s6_temp", "s7_temp", "s8_temp",
    #  "s1_volt", "s2_volt", "s3_volt", "s4_volt", "s5_volt", "s6_volt", "s7_volt", "s8_volt",
]

KEYS_OPTITRACK = [
    "ANT", "FOOD"
]

KEYS_FROM_ROBOT = KEYS_MOTORS + KEYS_BATTERY


def get_scaling_dict():
    d = defaultdict(lambda: 1)
    d["temperature"] = 10
    d["pitch"] = 100
    d["roll"] = 100
    d["yaw"] = 100
    d["accel_x"] = 100
    d["accel_y"] = 100
    d["accel_z"] = 100
    d["gyro_x"] = 100
    d["gyro_y"] = 100
    d["gyro_z"] = 100
    d["current"] = 100
    for s in ["s1_volt", "s2_volt", "s3_volt", "s4_volt", "s5_volt", "s6_volt", "s7_volt", "s8_volt"]:
        d[s] = 10
    d["volt"] = 100
    d["watt"] = 100
    d["charge"] = 10
    return d


OBS_SCALING = get_scaling_dict()


# Special Commands
class SpecialAction(enum.Enum):
    SHUTDOWN = "req:shutdown"
    RESET = "req:reset"
    CLOSURE = "req:closure"
    ATTACH = "req:attach"
    DETACH = "req:detach"
    RESET_BATTERY_INFO = "req:reset_bat_info"


SERVO_ID_TO_ACTION_DIM = {
    0: 6,
    1: 7,
    2: 4,
    3: 5,
    4: 2,
    5: 3,
    6: 0,
    7: 1,
}

FLAT_POSTURE_ACTION = np.array([0, -1, 0, 1, 0, 1, 0, -1], dtype=np.float32)
