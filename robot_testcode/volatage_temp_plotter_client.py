import json
import math
import os
from collections import OrderedDict

import numpy as np
import rel
import websocket  # websocket-client

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


try:
    import thread
except ImportError:
    import _thread as thread
import time


time_tick = []
temp_cpu_hist = []
temp_hist = []
volt_hist = []
ave_volt_hist = []


def encode_action_message(action):
    range_ = 0.5 * (512 - 400)
    half = range_ + 400
    scaled_action = range_ * action + half
    
    encoded_action = OrderedDict({i: int(scaled_action[i]) for i in range(8)})
    
    encoded_action = json.dumps(encoded_action)
    
    return encoded_action


def decode_message(message):
    if message == "Hello! World!":
        return None, False

    try:
        m = json.loads(message)
    # print("receive : {}".format(m))
    except json.JSONDecodeError as e:
        print(e)

    sensor_data = [m[k] if k == "id" else int(m[k]) for k in m.keys()]
    # print("raw sensor data: ", sensor_data)

    sensor_data = np.array(sensor_data[0:38] + sensor_data[39:], dtype=np.float32)
    
    print("sensor data: ", sensor_data, ", size: ", sensor_data.shape)

    clock = sensor_data[0]
    angle = sensor_data[1:9]
    sp = sensor_data[9:17] - angle
    feet = sensor_data[17:21]  # zero all
    temp_motor = sensor_data[21:29]
    volt_motor = sensor_data[29:37] / 10.
    temp = [sensor_data[38] / 10.]
    humid = [sensor_data[39]]
    pressure = [sensor_data[40]]
    orientation = sensor_data[41:44] / 100.
    accel = sensor_data[44:47] / 100.
    gyro = sensor_data[47:50] / 100.

    return temp_motor, volt_motor, temp


class WebsocketClient:

    def __init__(self, host_addr):
        self.host_addr = host_addr

        websocket.enableTrace(False)

        self.ws = websocket.WebSocketApp(
            url=host_addr,
            on_message=lambda ws, msg: self.on_message(ws, msg),
            on_error=lambda ws, msg: self.on_error(ws, msg),
            on_close=lambda ws: self.on_close(ws),
            on_open=lambda ws: self.on_open(ws)
        )

        self.latest_temp = None
        self.latest_motor_volt = None
        self.latest_motor_temp = None

    def on_message(self, ws, message):
        # print("message: ", message)
        self.latest_motor_temp, self.latest_motor_volt, self.latest_temp = decode_message(message)

    def on_error(self, ws, msg):
        print("On Error! ", time.time(), msg)

    def on_close(self, ws):
        print("### closed ###")

    def on_open(self, ws):
        thread.start_new_thread(self.run, ())

    def run(self, *args):
        global temp_cpu_hist, temp_hist, volt_hist, ave_volt_hist, time_tick

        scale = np.ones(8)
        time_sleep = 1
        step = 1
        while True:
            print("step @ ", step)
            
            if len(temp_hist) > 0:
                for i in range(8):
                    if temp_hist[-1][i] > 65:
                        scale[i] = 0.0
                    elif temp_hist[-1][i] < 50:
                        scale[i] = 1.0
                
            action = scale * np.random.uniform(-1, 1, 8)
            
            s = encode_action_message(action)

            # if step % 20 == 0:
            #     s = "req:shutdown"
            #     time_sleep = 60
            # else:
            #     s = "knock knock!"
            #     time_sleep = 1

            print("send message: ", s)

            try:
                self.ws.send(s)
            except websocket.WebSocketConnectionClosedException as e:
                print("Error!: ", e)
                break
                
            if not any([self.latest_temp is None, self.latest_motor_temp is None, self.latest_motor_volt is None]):
                
                time_tick.append(time.time())
                
                temp_cpu_hist.append(self.latest_temp)
                temp_hist.append(self.latest_motor_temp)
                
                if self.latest_motor_volt is not None:
                    volt_hist.append(self.latest_motor_volt)
                    
                if self.latest_motor_volt is not None:
                    ave_volt_hist.append(np.mean(self.latest_motor_volt))
                    
                # print("sensors: ", self.latest_temp, self.latest_motor_temp, self.latest_temp)
            
            step += 1

            time.sleep(time_sleep)

        print("web socket thread ended.")

    # websocketクライアント起動
    def run_forever(self):
        self.ws.run_forever()


HOST_ADDR = "ws://192.168.3.20:9001/"
ws_client = WebsocketClient(HOST_ADDR)
thread.start_new_thread(ws_client.ws.run_forever, ())

# while True:
#     try:
#         ws_client.ws.run_forever(dispatcher=rel, reconnect=3)
#         rel.signal(2, rel.abort)  # Keyboard Interrupt
#         rel.dispatch()
#     except ConnectionResetError as e:
#         ws_client = WebsocketClient(HOST_ADDR)
#         print(e)

start_at = time.time()

plt.figure()

while True:
    plt.clf()
    
    t_ = np.array(time_tick) - start_at
    
    if len(t_) > 5:
        
        plt.subplot(311)
        y_ = np.stack(temp_hist, axis=0).transpose()
        for i in range(8):
            plt.plot(y_[i], alpha=0.5)

        plt.ylim([20.0, 80.0])
        plt.title("motor temp")
        plt.xlabel("approx. 1 sec")
        plt.ylabel("[degree]")

        plt.subplot(312)
        y_ = np.stack(volt_hist, axis=0).transpose()
        for i in range(8):
            plt.plot(y_[i], linewidth=1, alpha=0.5)
    
        plt.plot(ave_volt_hist, "r", linewidth=3)
        plt.ylim([9.0, 13])
        plt.title("motor voltage")
        plt.xlabel("approx. 1 sec")
        plt.ylabel("[V]")

        plt.subplot(313)
        plt.plot(temp_cpu_hist)
        plt.title("CPU temp")
        plt.xlabel("approx. 1 sec")
        plt.ylabel("[C]")
        
        plt.tight_layout()

        plt.pause(0.0001)
        
        os.makedirs("volt_temp_data", exist_ok=True)
        np.save("volt_temp_data/volt_hist.npy", volt_hist)
        np.save("volt_temp_data/ave_volt_hist.npy", ave_volt_hist)
        np.save("volt_temp_data/temp_cpu_hist.npy", temp_cpu_hist)
        np.save("volt_temp_data/temp_hist.npy", temp_hist)
        
        print("saved.")

    time.sleep(0.3)
