import json
import logging
import time

import threading
from collections import OrderedDict

import ipget
import smbus2

from websocket_server import WebsocketServer  # pip install git+https://github.com/Pithikos/python-websocket-server

from const import DEVICE_ADDR, CTRL_ADDR, CURRENT_ADDR, VOLT_ADDR, WATT_ADDR, CHARGE_ADDR, ENERGY_ADDR, KEYS_BATTERY
from utils import decode_battery_data

# i2c
i2c = smbus2.SMBus(1)
i2c.write_byte_data(DEVICE_ADDR, CTRL_ADDR, 0b0000_1000)

batcnt = 0
latest_battery_val = OrderedDict()


def read_battery_measurements(i2c):
    global latest_battery_val, batcnt
    
    while True:
        current = i2c.read_i2c_block_data(DEVICE_ADDR, CURRENT_ADDR, 3)
        current = 3e-3 * decode_battery_data(current, 3)
        
        volt = i2c.read_i2c_block_data(DEVICE_ADDR, VOLT_ADDR, 2)
        volt = 2e-3 * decode_battery_data(volt, 2)
        
        watt = i2c.read_i2c_block_data(DEVICE_ADDR, WATT_ADDR, 3)
        watt = 5e-3 * decode_battery_data(watt, 3)
        
        charge = i2c.read_i2c_block_data(DEVICE_ADDR, CHARGE_ADDR, 6)
        charge = 1.193e-6 * decode_battery_data(charge, 6)
        
        energy = i2c.read_i2c_block_data(DEVICE_ADDR, ENERGY_ADDR, 6)
        energy = 19.89e-6 * decode_battery_data(energy, 6)
        
        vals = [
            int(100 * current),
            int(100 * volt),
            int(100 * watt),
            int(10 * charge),
            int(energy)
        ]

        bat_val = OrderedDict(map(lambda i: (i[1], vals[i[0]]), list(enumerate(KEYS_BATTERY))))

        if batcnt % 10 == 0:
            # print(line.strip("\r\n"))
            print(bat_val)
            batcnt = 0
        batcnt += 1
        
        latest_battery_val = bat_val
        

class WebsocketServerHandler:
    
    def __init__(self, host, port):
        self.latest_sensor = None
        
        self.server = WebsocketServer(port=port, host=host, loglevel=logging.DEBUG)
        
        self.step = 0
    
    def new_client(self, client, server):
        print(client, server)
    
    def client_left(self, client, server):
        pass
    
    def message_received(self, client, server, message):
        global latest_battery_val, i2c
        
        print(client["id"], "send message: ", message)
        
        if message == "req:measurement":
            self.server.send_message_to_all(json.dumps(latest_battery_val))
            
        elif message == "req:reset_bat_info":
            i2c.write_byte_data(DEVICE_ADDR, CTRL_ADDR, 0b0000_0010)
            i2c.write_byte_data(DEVICE_ADDR, CTRL_ADDR, 0b0000_1000)
            time.sleep(1.5)
            
    def run(self):
        global latest_battery_val
        
        time.sleep(0.1)
        
        self.server.set_fn_new_client(self.new_client)
        
        self.server.set_fn_client_left(self.client_left)
        
        self.server.set_fn_message_received(self.message_received)
        
        self.server.run_forever()


IP_ADDR = ipget.ipget().ipaddr("eth0").split("/")[0]
PORT = 9001
ws_server = WebsocketServerHandler(IP_ADDR, PORT)

battery_thread = threading.Thread(target=read_battery_measurements, args=(i2c,))
battery_thread.start()

time.sleep(1)

try:
    ws_server.run()
finally:
    i2c.close()
