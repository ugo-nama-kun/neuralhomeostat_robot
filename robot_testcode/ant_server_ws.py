import sys
import threading

from websocket_server import WebsocketServer  # pip install git+https://github.com/Pithikos/python-websocket-server
import logging
import zmq
import time
import numpy as np
import serial
from collections import OrderedDict
import datetime
import json


# Serial
connected = False
ser = serial.Serial('/dev/ttyACM0', 115200)

valcnt = 0
latest_sensor_val = OrderedDict()


def read_measurements(ser):
    global connected,sock,valcnt,latest_sensor_val
    while not connected:
        connected = True
        last_ant_time = 0
        while True:
            line = ser.readline().decode()
            if line.startswith("- recv"):
                pass
                #print(line.strip("\r\n"))

            elif line.startswith("meas "):
                vals = line[5:].split("\t")
                keys = ["ant_time", "s1_angle", "s2_angle", "s3_angle", "s4_angle", "s5_angle", "s6_angle", "s7_angle", "s8_angle", "s1_sp", "s2_sp", "s3_sp", "s4_sp", "s5_sp", "s6_sp", "s7_sp", "s8_sp", "feet1", "feet2", "feet3", "feet4", "s1_temp", "s2_temp", "s3_temp", "s4_temp", "s5_temp", "s6_temp", "s7_temp", "s8_temp"]
                if len(vals) >= len(keys):
                    latest_sensor_val = OrderedDict(map(lambda i: (i[1],vals[i[0]]), list(enumerate(keys))))
                    if valcnt % 50 == 0:
                        print(line.strip("\r\n"))
                    valcnt += 1

                    if float(latest_sensor_val["ant_time"]) - last_ant_time < 0:
                        print("implausible ant time, skipping", latest_sensor_val)
                        continue

                    last_ant_time = float(latest_sensor_val["ant_time"])

                    utcnowms = int((datetime.datetime.utcnow() - datetime.datetime(1970, 1, 1)).total_seconds() * 1000)
                    #print(utcnowms)
                    latest_sensor_val["server_epoch_ms"] = utcnowms
                    latest_sensor_val["id"] = "serial"
                    #sock.send_json(latest_sensor_val)
            else:
                if line.strip("\r\n ") != "":
                    print(line.strip("\r\n"))


class Websocket_Server():
    
    def __init__(self, host, port):
        self.latest_sensor = None
        
        self.server = WebsocketServer(port=port, host=host, loglevel=logging.DEBUG)
        
    def new_client(self, client, server):
        print(client, server)
        self.server.send_message_to_all("Hello! World!")
        
    def client_left(self, client, server):
        global ser
        
        print(client, "has left.")
        
        time.sleep(1)
        ser.write(b"reset\n")
        time.sleep(1)
        ser.write(b"detach_servos\n")
        
    def message_received(self, client, server, message):
        global ser, latest_sensor_val
        
        print(client["id"], "send message: ", message)

        cmd = json.loads(message)
        m = [cmd[k] for k in cmd.keys()]

        s = f"s1 {m[0]} s2 {m[1]} s3 {m[2]} s4 {m[3]} s5 {m[4]} s6 {m[5]} s7 {m[6]} s8 {m[7]}\n"
        s = bytes(s.encode("utf-8"))
    
        ser.write(s)
        
        data = latest_sensor_val

        print("data: ", data["s1_angle"], ", ", data["s1_sp"])

        self.server.send_message_to_all(json.dumps(data))
        #self.server.send_message_to_all("getcha!")
        
    # def get_sensor_val(self):
    #     global connected,valcnt
    #
    #     data = None
    #
    #     last_ant_time = 0
    #
    #     while data is None:
    #
    #         line = ser.readline().decode()
    #
    #         if line.startswith("- recv"):
    #             pass
    #             #print(line.strip("\r\n"))
    #
    #         elif line.startswith("meas "):
    #             vals = line[5:].split("\t")
    #             keys = ["ant_time", "s1_angle", "s2_angle", "s3_angle", "s4_angle", "s5_angle", "s6_angle", "s7_angle", "s8_angle", "s1_sp", "s2_sp", "s3_sp", "s4_sp", "s5_sp", "s6_sp", "s7_sp", "s8_sp", "feet1", "feet2", "feet3", "feet4", "s1_temp", "s2_temp", "s3_temp", "s4_temp", "s5_temp", "s6_temp", "s7_temp", "s8_temp"]
    #             if len(vals) >= len(keys):
    #                 data = OrderedDict(map(lambda i: (i[1],vals[i[0]]), list(enumerate(keys))))
    #                 if valcnt % 50 == 0:
    #                     print(line.strip("\r\n"))
    #                 valcnt += 1
    #
    #                 if float(data["ant_time"]) - last_ant_time < 0:
    #                     print("implausible ant time, skipping", data)
    #                     continue
    #
    #                 last_ant_time = float(data["ant_time"])
    #
    #                 utcnowms = int((datetime.datetime.utcnow() - datetime.datetime(1970, 1, 1)).total_seconds() * 1000)
    #                 #print(utcnowms)
    #                 data["server_epoch_ms"] = utcnowms
    #                 data["id"] = "serial"
    #
    #                 #print(data)
    #         else:
    #             if line.strip("\r\n ") != "":
    #                 print(line.strip("\r\n"))
    #
    #     return data

    def run(self):
        global ser
        
        time.sleep(0.1)
        
        ser.write(b"attach_servos\n")
        
        time.sleep(0.1)
        
        self.server.set_fn_new_client(self.new_client)
        
        self.server.set_fn_client_left(self.client_left)
        
        self.server.set_fn_message_received(self.message_received)
        
        self.server.run_forever()
        

IP_ADDR = "192.168.3.20"
PORT = 9001
ws_server = Websocket_Server(IP_ADDR, PORT)

thread = threading.Thread(target=read_measurements, args=(ser,))
thread.start()

time.sleep(2)

ws_server.run()
