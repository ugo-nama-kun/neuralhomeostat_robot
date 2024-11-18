from collections import OrderedDict

import numpy as np
import websocket  #  websocket-client
import json

try:
	import thread
except ImportError:
	import _thread as thread
import time


class Websocket_Client():

	def __init__(self, host_addr):
		# デバックログの表示/非表示設定
		websocket.enableTrace(False)

		# WebSocketAppクラスを生成
		# 関数登録のために、ラムダ式を使用
		self.ws = websocket.WebSocketApp(
			url=host_addr,
			on_message=lambda ws, msg: self.on_message(ws, msg),
			on_error=lambda ws, msg: self.on_error(ws, msg),
			on_close=lambda ws: self.on_close(ws),
		)

		self.ws.on_open = lambda ws: self.on_open(ws)

	# メッセージ受信に呼ばれる関数
	def on_message(self, ws, message):
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
		print("angle:", angle)
		print("speed:", sp - angle)
		# print("temp:", temp)


	# エラー時に呼ばれる関数
	def on_error(self, ws, error):
		print(error)

	# サーバーから切断時に呼ばれる関数
	def on_close(self, ws):
		print("### closed ###")

	# サーバーから接続時に呼ばれる関数
	def on_open(self, ws):
		thread.start_new_thread(self.run, ())

	# サーバーから接続時にスレッドで起動する関数
	def run(self, *args):
		for i in range(10):
			input_data = OrderedDict({
				0: np.random.randint(400, 512),
				1: np.random.randint(200, 312),
				2: np.random.randint(400, 512),
				3: np.random.randint(200, 312),
				4: np.random.randint(400, 512),
				5: np.random.randint(200, 312),
				6: np.random.randint(400, 512),
				7: np.random.randint(200, 312),
			})

			# input_data = OrderedDict({
			# 	0: 450,
			# 	1: 300 + 20 * i,
			# 	2: 450,
			# 	3: 450,
			# 	4: 450,
			# 	5: 450,
			# 	6: 450,
			# 	7: 450,
			# })
			self.ws.send(json.dumps(input_data))
			time.sleep(0.1)

		self.ws.close()
		print("thread terminating...")

	# websocketクライアント起動
	def run_forever(self):
		self.ws.run_forever()


HOST_ADDR = "ws://192.168.3.20:9001/"
ws_client = Websocket_Client(HOST_ADDR)
ws_client.run_forever()
