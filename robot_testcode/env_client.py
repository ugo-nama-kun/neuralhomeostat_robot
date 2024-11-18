import json
import time

import numpy as np
import zmq
from numpy._typing import NDArray

from const import OBS_SCALING, KEYS_FROM_ROBOT
from utils import action2servo

context = zmq.Context()

socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")


def encode_action_message(action: NDArray) -> bytes:
    m = action2servo(action)
    
    s = f"s1 {m[0]} s2 {m[1]} s3 {m[2]} s4 {m[3]} s5 {m[4]} s6 {m[5]} s7 {m[6]} s8 {m[7]}\n"
    s = bytes(s.encode("utf-8"))
    
    return s

dt = 0

for i in range(10):
    last_time = time.time()

    action = 0.1 * np.random.uniform(-1, 1, 8)
    
    action_message = encode_action_message(action)
    socket.send(action_message)
    
    obs_dict = json.loads(socket.recv())

    # Sensor val scaling
    obs_dict.update({k: int(obs_dict[k]) / OBS_SCALING[k] for k in KEYS_FROM_ROBOT})
    print(f"Received replay {i}: {obs_dict}")
    
    dt = dt + 0.2 * ((time.time() - last_time) - dt)
    print(dt)


socket.send(b"reset")
obs_message = json.loads(socket.recv())
print(f"Received replay Final: {obs_message}")
