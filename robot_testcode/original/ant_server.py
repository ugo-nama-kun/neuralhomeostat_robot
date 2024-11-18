# Licensed under MIT licence, see LICENSE for details.
# Copyright Ote Robotics Ltd. 2020

import serial
import time
import threading
import json
from collections import OrderedDict
import zmq
import datetime

import matplotlib.pyplot as plt

ctx = zmq.Context()
sock = ctx.socket(zmq.PUB)
sock.connect("tcp://127.0.0.1:3002")
cmdsock = ctx.socket(zmq.SUB)
cmdsock.connect("tcp://127.0.0.1:3001")
cmdsock.setsockopt_string(zmq.SUBSCRIBE, "cmd")

connected = False
ser = serial.Serial('/dev/ttyACM0', 115200)
#ser = serial.Serial('/dev/cu.ANT-HC-05-DevB', 115200)
#ser = serial.Serial('/dev/rfcomm0', 115200)
#ser = serial.Serial('/dev/cu.usbmodem1442201', 115200)

valcnt = 0
data_temp = [[] for _ in range(8)]
data_voltage = [[] for _ in range(8)]
time_stamp = []


def read_measurements(ser):
    global connected,sock,valcnt,data_voltage

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
                keys = [
                    "ant_time",
                    "s1_angle", "s2_angle", "s3_angle", "s4_angle", "s5_angle", "s6_angle", "s7_angle", "s8_angle",
                    "s1_sp", "s2_sp", "s3_sp", "s4_sp", "s5_sp", "s6_sp", "s7_sp", "s8_sp",
                    "feet1", "feet2", "feet3", "feet4",
                    "s1_temp", "s2_temp", "s3_temp", "s4_temp", "s5_temp", "s6_temp", "s7_temp", "s8_temp",
                    "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
                ]
                # print("---")
                # print(line)
                # print(len(vals), len(keys))
                if len(vals) >= len(keys):
                    d = OrderedDict(map(lambda i: (i[1],vals[i[0]]), list(enumerate(keys))))
                    if valcnt % 50 == 0:
                        print(line.strip("\r\n"))
                        time_stamp.append(time.time())
                        for i in range(8):
                            data_voltage[i].append(int(d[f"v{i+1}"]))
                            data_temp[i].append(int(d[f"s{i+1}_temp"]))
                    valcnt += 1

                    if float(d["ant_time"]) - last_ant_time < 0:
                        print("implausible ant time, skipping", d)
                        continue

                    last_ant_time = float(d["ant_time"])

                    utcnowms = int((datetime.datetime.utcnow() - datetime.datetime(1970, 1, 1)).total_seconds() * 1000) 
                    #print(utcnowms)
                    d["server_epoch_ms"] = utcnowms
                    d["id"] = "serial"
                    sock.send_json(d)

            else:
                if line.strip("\r\n ") != "":
                    print(line.strip("\r\n"))


thread = threading.Thread(target=read_measurements, args=(ser,))
thread.start()

time.sleep(2)

plt.figure()

while True:
    c = cmdsock.recv_multipart()
    if c[1][0] != b's':
       print(c, c[1])
    ser.write(c[1])

    plt.clf()
    plt.subplot(211)
    for i in range(8):
        plt.plot(time_stamp, data_voltage[i], alpha=0.5)
    plt.title("voltage")
    plt.xlabel("Step (approx 1 sec)")
    plt.ylabel("Voltage [x 10 V]")

    plt.subplot(212)
    for i in range(8):
        plt.plot(time_stamp, data_temp[i], alpha=0.5)
    plt.title("temperature")
    plt.xlabel("Step (approx 1 sec)")
    plt.ylabel("Temperature [x 10 C]")

    plt.tight_layout()
    plt.pause(0.00001)

    print(data_voltage)
