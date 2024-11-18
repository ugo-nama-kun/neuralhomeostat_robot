import json
import threading
import time

import cv2
import gym
import numpy as np
import tqdm as tqdm
import websocket  # websocket-client

from collections import OrderedDict
from gym import error, spaces, utils
from gym.utils import seeding


def decode_message(message):
	if message == "Hello! World!":
		return None, False

	try:
		m = json.loads(message)
	# print("receive : {}".format(m))
	except json.JSONDecodeError as e:
		print(e)

	sensor_data = [m[k] if k == "id" else int(m[k]) for k in m.keys()]

	print(sensor_data)

	sensor_data = np.array(sensor_data[0:30] + sensor_data[31:])

	clock = sensor_data[0]
	angle = sensor_data[1:9]
	sp = sensor_data[9:17] - angle
	feet = sensor_data[17:21]  # zero all
	temp_motor = sensor_data[21:29]
	temp = [sensor_data[30] / 10.]
	humid = [sensor_data[31]]
	pressure = [sensor_data[32]]
	orientation = sensor_data[33:36] / 100.
	accel = sensor_data[36:39] / 100.
	gyro = sensor_data[39:42] / 100.

	return np.concatenate([
		angle,
		sp,
		temp_motor,
		temp,
		humid,
		pressure,
		orientation,
		accel,
		gyro,
	]).astype(np.float32), True


class RealAntEnvOld(gym.Env):
	metadata = {'render.modes': ['human']}

	HOST_ADDR = "ws://192.168.3.20:9001/"
	URL = "http://192.168.3.20:8080/?action=stream"

	# HOST_ADDR = "ws://172.16.1.1:9001/"
	# URL = "http://172.16.1.1:8080/?action=stream"

	# HOST_ADDR = "ws://10.42.0.1:9001/"
	# URL = "http://10.42.0.1:8080/?action=stream"

	def __init__(self, addr=None):
		self.addr = addr

		websocket.enableTrace(False)

		self.ws = websocket.create_connection(self.HOST_ADDR if self.addr is None else self.addr)

		self.action_space = spaces.Box(low=-1, high=1, shape=(8,))

		self.target_fps = 10.
		self.target_duration = 1 / self.target_fps  # 10Hz

		self.start_at = time.time()

	def step(self, action, request_shutdown=False):
		self.send_action(action)

		dt = time.time() - self.start_at
		duration = max((0, self.target_duration - dt))
		time.sleep(duration)

		if request_shutdown:
			self.send_shutdown()

			self.ws.close()

			print("Waiting for reboot.")
			for _ in tqdm.tqdm(range(120)):
				time.sleep(1)  # TODO use flag?

			success = False
			while not success:
				try:
					self.ws = websocket.create_connection(
						self.HOST_ADDR if self.addr is None else self.addr,
					)
					success = True
					print("Connection Success")
				except ConnectionRefusedError as e:
					print("Retry Connection...")
					time.sleep(0.3)

		self.start_at = time.time()

		obs = self.get_obs()
		reward = 0
		done = False
		info = {}

		return obs, reward, done, info

	def send_action(self, action):
		encoded_action = self.encode_action_message(action)
		self.ws.send(encoded_action)

	def encode_action_message(self, action):
		range_ = 0.5 * (512 - 400)
		half = range_ + 400
		scaled_action = range_ * action + half

		encoded_action = OrderedDict({i: int(scaled_action[i]) for i in range(8)})

		encoded_action = json.dumps(encoded_action)

		return encoded_action

	def send_shutdown(self):
		self.ws.send("req:shutdown")

	def get_obs(self):
		vec_data = None

		success = False
		while not success:
			result = self.ws.recv()
			vec_data, success = decode_message(result)

		# get image
		video = cv2.VideoCapture(self.URL)
		ret, img = video.read()
		video.release()

		obs = {
			"vector": vec_data,
			"img"   : cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
		}

		# print("sensor: ", obs)

		return obs

	def reset(self):
		self.ws.close()
		time.sleep(0.1)
		self.ws = websocket.create_connection(self.HOST_ADDR if self.addr is None else self.addr)
		time.sleep(1)
		self.send_action(action=np.ones(self.action_space.shape))
		time.sleep(1)
		self.start_at = time.time()
		return self.get_obs()

	def close(self):
		self.ws.send("req:closure")
		self.ws.close()

	def render(self, mode="human"):
		video = cv2.VideoCapture(self.URL)
		ret, img = video.read()

		img = cv2.resize(img, dsize=(320, 320), interpolation=cv2.INTER_NEAREST)

		cv2.imshow("Test", img)

		cv2.waitKey(1)
