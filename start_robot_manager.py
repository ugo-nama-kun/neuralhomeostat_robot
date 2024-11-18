import datetime
import json
import platform
import threading
import time
from collections import OrderedDict

import serial
import zmq
import websocket
import natnet

from const import TARGET_DURATION, KEYS_MOTORS, KEYS_OPTITRACK
from network import IP_RASPI, MAC_PORT, IP_OPTITRACK


# ID should be same with streaming ID in Motive Rigid body
OBJECT_NAMES = {
    1: KEYS_OPTITRACK[0],  # ANT
    2: KEYS_OPTITRACK[1],  # FOOD
}


class ExperimentManager:
    
    # Raspberry Pi
    HOST_ADDR = f"ws://{IP_RASPI}:9001/"
    
    def __init__(self, detach_finish):
        # TODO: add function to partially disable functions
        self.enable_optitrack_meas = True
        self.enable_battery_meas = True
        self.detach_finish = detach_finish
        
        # Arduino 33 IOT
        context = zmq.Context()
        self.socket = context.socket(zmq.REP)
        self.socket.bind("tcp://*:5555")

        self.ser = None
        if platform.system() == "Darwin":
            # MacOS
            self.ser = serial.Serial(MAC_PORT, 115200)
        else:
            # Linux
            self.ser = serial.Serial('/dev/ttyACM0', 115200)
            
        # Websocket (Raspi)
        self.ws = None
        if self.enable_battery_meas:
            websocket.enableTrace(False)
            self.ws = websocket.WebSocket()

        # OptiTrack server
        if self.enable_optitrack_meas:
            self.optitrack_client = natnet.Client.connect(server=IP_OPTITRACK)
        
        self.connected = False
        self.valcnt = 0
        
        self.motor_measurement = None
        self.battery_measurement = None
        self.optitrack_measurement = None

    def run_forever(self):
        self._setup()
        
        self.ser.write(b"attach_servos\n")
        time.sleep(0.5)
        # self.ser.write(b"reset\n")
        # time.sleep(1)
        
        try:
            while True:
                last_time = datetime.datetime.utcnow()
                
                message_action = self.socket.recv()
                
                # applying action
                self.ser.write(message_action)
                
                # Battery info request
                if self.enable_battery_meas:
                    self.ws.send("req:measurement")
                    self.battery_measurement = json.loads(self.ws.recv())
                
                # Wait for the fixed interval
                time.sleep(max(TARGET_DURATION - (datetime.datetime.utcnow() - last_time).total_seconds(), 0))
                
                print(f"battery: {self.battery_measurement}")
                print(f"motor: {self.motor_measurement}")
                print(f"objects: {self.optitrack_measurement}")
                
                try:
                    measurement = dict(self.motor_measurement)
                    if self.enable_battery_meas:
                        measurement.update(self.battery_measurement)
                    if self.enable_optitrack_meas:
                        measurement.update(self.optitrack_measurement)
                    
                    measurement = json.dumps(measurement).encode("ascii")

                    # send measurement
                    self.socket.send(measurement)
                except TypeError as e:
                    print(f"Error Handling. : {e}")
                
                print("---")
        
        finally:
            if self.detach_finish:
                # self.ser.write(b"reset\n")
                # time.sleep(1)
                self.ser.write(b"detach_servos\n")
            
            self.socket.close()
            self.ser.close()
            if self.ws:
                self.ws.close()
    
    def _setup(self):
        thread = threading.Thread(target=self.read_motor_measurements, args=(self.ser,))
        thread.setDaemon(True)
        thread.start()
        
        if self.enable_battery_meas:
            self.ws.connect(self.HOST_ADDR)
        
        def callback(rigid_bodies, markers, timing):
            # print(rigid_bodies)
            self.optitrack_measurement = {
                OBJECT_NAMES[rb.id_]: (rb.position, rb.orientation) for rb in rigid_bodies
            }
        
        if self.enable_optitrack_meas:
            self.optitrack_client.set_callback(callback)
            optitrack_thread = threading.Thread(target=self.optitrack_client.spin)
            optitrack_thread.setDaemon(True)
            optitrack_thread.start()
    
    def read_motor_measurements(self, ser):
        while not self.connected:
            self.connected = True
            last_ant_time = 0
            while True:
                line = ser.readline().decode()
                if line.startswith("meas "):
                    vals = line[5:].split("\t")
                    if len(vals) >= len(KEYS_MOTORS):
                        d = OrderedDict(map(lambda i: (i[1], vals[i[0]]), list(enumerate(KEYS_MOTORS))))
                        # if valcnt % 50 == 0:
                        #     print(line.strip("\r\n"))
    
                        self.valcnt += 1
                        
                        if float(d["ant_time"]) - last_ant_time < 0:
                            print("implausible ant time, skipping", d)
                            continue
                        
                        last_ant_time = float(d["ant_time"])
                        
                        utcnowms = int((datetime.datetime.utcnow() - datetime.datetime(1970, 1, 1)).total_seconds() * 1000)
                        d["server_epoch_ms"] = utcnowms
                        d["id"] = "serial"
                        
                        self.motor_measurement = d
                
                else:
                    if line.strip("\r\n ") != "":
                        print(line.strip("\r\n"))


if __name__ == '__main__':
    import argparse
    from distutils.util import strtobool
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="detach servo when close environment.")
    args = parser.parse_args()

    manager = ExperimentManager(detach_finish=args.d)
    manager.run_forever()
