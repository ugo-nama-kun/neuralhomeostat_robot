import collections
import itertools
import math

import numpy as np

from const import KEYS_JOINT, hip_range, hip_offset, ankle_range, ankle_offset, SERVO_ID_TO_ACTION_DIM


def lerp(a, b, t: float):
    # Linear interpolate [-1, 1] -> [a, b]
    t_ = 0.5 * (t + 1)
    return (1 - t_) * a + t_ * b


def unlerp(a, b, y: float):
    # Inverse of lerp function
    return 2 * ((y - a) / (b - a)) - 1


def action2servo(action: np.ndarray):
    # adjust ordering, range and offsets for the physical ant
    # from https://github.com/AaltoVision/realant-rl/blob/3a5b4bbe7f48eaef9ee113e0eeb3355ebeeed547/rollout_server.py#L188
    servo = np.zeros(8, dtype=np.int64)
    servo[6] = lerp(hip_offset + hip_range, hip_offset - hip_range, -action[0])
    servo[7] = lerp(ankle_offset + ankle_range, ankle_offset - ankle_range, action[1])
    servo[4] = lerp(hip_offset + hip_range, hip_offset - hip_range, -action[2])
    servo[5] = lerp(ankle_offset + ankle_range, ankle_offset - ankle_range, -action[3])
    servo[2] = lerp(hip_offset + hip_range, hip_offset - hip_range, -action[4])
    servo[3] = lerp(ankle_offset + ankle_range, ankle_offset - ankle_range, -action[5])
    servo[0] = lerp(hip_offset + hip_range, hip_offset - hip_range, -action[6])
    servo[1] = lerp(ankle_offset + ankle_range, ankle_offset - ankle_range, action[7])
    
    return servo


def action2angle(action: np.ndarray):  # ok
    """
    Converting action to target angles
    :param action:
    :return:
    """
    joint_angles = np.zeros_like(action)
    joint_angles[0] = lerp(-0.69, 0.69, action[0])  # hip_1
    joint_angles[1] = lerp(0.0, 1.05, action[1])  # ankle_1
    joint_angles[2] = lerp(-0.69, 0.69, action[2])  # hip_2
    joint_angles[3] = lerp(-1.05, 0.0, action[3])  # ankle_2
    joint_angles[4] = lerp(-0.69, 0.69, action[4])  # hip_3
    joint_angles[5] = lerp(-1.05, 0.0, action[5])  # ankle_3
    joint_angles[6] = lerp(-0.69, 0.69, action[6])  # hip_4
    joint_angles[7] = lerp(0.0, 1.05, action[7])  # ankle_4
    return joint_angles


def servo2action(servo_val):
    action = np.zeros_like(servo_val, dtype=np.float32)
    action[0] = -unlerp(hip_offset + hip_range, hip_offset - hip_range, servo_val[SERVO_ID_TO_ACTION_DIM[0]])
    action[1] = unlerp(ankle_offset + ankle_range, ankle_offset - ankle_range, servo_val[SERVO_ID_TO_ACTION_DIM[1]])
    action[2] = -unlerp(hip_offset + hip_range, hip_offset - hip_range, servo_val[SERVO_ID_TO_ACTION_DIM[2]])
    action[3] = -unlerp(ankle_offset + ankle_range, ankle_offset - ankle_range, servo_val[SERVO_ID_TO_ACTION_DIM[3]])
    action[4] = -unlerp(hip_offset + hip_range, hip_offset - hip_range, servo_val[SERVO_ID_TO_ACTION_DIM[4]])
    action[5] = -unlerp(ankle_offset + ankle_range, ankle_offset - ankle_range, servo_val[SERVO_ID_TO_ACTION_DIM[5]])
    action[6] = -unlerp(hip_offset + hip_range, hip_offset - hip_range, servo_val[SERVO_ID_TO_ACTION_DIM[6]])
    action[7] = unlerp(ankle_offset + ankle_range, ankle_offset - ankle_range, servo_val[SERVO_ID_TO_ACTION_DIM[7]])
    return action


def get_joint(obs_dict: dict):
    servo_val = np.array([obs_dict[k] for k in KEYS_JOINT], dtype=np.float32)

    action = servo2action(servo_val)  # decoding
    jpos = action2angle(action)

    # 0 hip left forward
    # 1 ankle left forward
    # 2 hip left back
    # 3 ankle left back
    # 4 hip right back
    # 5 ankle right back
    # 6 hip right forward
    # 7 ankle right forward

    return jpos


def get_motor_temperature(obs_dict: dict):
    temp = list(map(lambda x: obs_dict[f"s"+str(SERVO_ID_TO_ACTION_DIM[x]+1)+"_temp"], np.arange(8)))
    return np.array(temp)

    
def get_orientation(obs_dict: dict):
    # TODO: remove yaw signal
    return np.array([obs_dict["yaw"], obs_dict["pitch"], obs_dict["roll"]])


# def get_accel(obs_dict: dict):
#     return np.array([obs_dict["accel_x"], obs_dict["accel_y"], obs_dict["accel_z"]], dtype=np.float32)
#
#
# def get_gyro(obs_dict: dict):
#     return np.array([obs_dict["gyro_x"], obs_dict["gyro_y"], obs_dict["gyro_z"]], dtype=np.float32)


def get_battery(obs_dict: dict):
    current = obs_dict["current"]
    volt = obs_dict["volt"]
    watt = obs_dict["watt"]
    charge = obs_dict["charge"]
    energy = obs_dict["energy"]
    return current, volt, watt, charge, energy


def decode_battery_data(data, byte, signed=True):
    out = 0
    
    for i in reversed(range(byte)):
        out |= data[i] << (8 * (byte - 1 - i))
        
    if signed:
        if (out & (1 << (8 * byte - 1))) != 0:
            out = out - (1 << (8 * byte))
            
    return out


class sliceable_deque(collections.deque):
    # from https://stackoverflow.com/questions/10003143/how-to-slice-a-deque
    def __getitem__(self, index):
        if isinstance(index, slice):
            return type(self)(itertools.islice(self, index.start,
                                               index.stop, index.step))
        return collections.deque.__getitem__(self, index)


def qtoeuler(q):
    """ quaternion to Euler angle

    :param q: quaternion
    :return:
    """
    from scipy.spatial.transform import Rotation
    
    # phi = math.atan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
    # theta = math.asin(2 * (q[0] * q[2] - q[3] * q[1]))
    # psi = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    # return np.array([psi, theta, phi])
    
    rot = Rotation.from_quat(q)
    
    # Convert the rotation to Euler angles given the axes of rotation
    return rot.as_euler('xyz', degrees=False)
