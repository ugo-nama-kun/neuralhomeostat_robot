import time
from collections import OrderedDict

import numpy as np
import websocket  #  websocket-client
import json


def decode_message(message):
	m = json.loads(message)
	# print("receive : {}".format(m))

	sensor_data = [m[k] if k == "id" else int(m[k]) for k in m.keys()]

	# print(sensor_data)

	sensor_data = np.array(sensor_data[0:-2])

	clock = sensor_data[0]
	angle = sensor_data[1:9]
	sp = sensor_data[9:17]
	feet = sensor_data[17:21]  # zero all
	temp = sensor_data[21:29]

	# print("clock:", clock)
	# print("angle:", angle)
	# print("speed:", sp - angle)
	# print("temp:", temp)

	return np.concatenate([angle, sp, temp])


def on_error(ws, error):
	print(error)


def on_close(ws):
	print("### closed ###")


if __name__ == '__main__':
	websocket.enableTrace(False)

	HOST_ADDR = "ws://192.168.3.20:9001/"

	ws = websocket.create_connection(HOST_ADDR)

	for i in range(10):
		action = np.random.randint(400, 512, 8).tolist()

		print("action:", action)

		input_data = OrderedDict({i: action[i] for i in range(8)})

		ws.send(json.dumps(input_data))

		time.sleep(0.1)

		result = ws.recv()
		while result == "Hello! World!":
			result = ws.recv()

		sensor_data = decode_message(result)

		print("sensor: ", sensor_data)

	ws.close()
	print("thread terminating...")
