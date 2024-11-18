import os

from websocket_server import WebsocketServer  # pip install git+https://github.com/Pithikos/python-websocket-server
import logging
import time

from sense_hat import SenseHat

sense = SenseHat()
sense.clear()


class Websocket_Server():

	def __init__(self, host, port):
		self.server = WebsocketServer(port=port, host=host, loglevel=logging.DEBUG)

	def new_client(self, client, server):
		print(client, server)

	def client_left(self, client, server):
		pass

	def message_received(self, client, server, message):
		print(client["id"], "send message: ", message)

		if message == "req:shutdown":
			sense.show_message(text_string="SHUTDOWN", text_colour=(100, 100, 100))
			time.sleep(5)
			os.system("sudo shutdown now -h")
			#os.system("sudo reboot now -h")  # For development purpose

			while True:
				sense.show_message(text_string="SHUTDOWN IN PROGRESS", text_colour=(100, 100, 100))

		elif message == "req:closure":
			sense.show_message(text_string="CLOSURE", text_colour=(100, 100, 100))
			time.sleep(1)

		else:
			s = "Hey!"
			s = bytes(s.encode("utf-8"))
			self.server.send_message_to_all(s)

	def run(self):
		self.server.set_fn_new_client(self.new_client)

		self.server.set_fn_client_left(self.client_left)

		self.server.set_fn_message_received(self.message_received)

		sense.show_message(text_string="REBOOTED", text_colour=(100, 100, 100))

		self.server.run_forever()


time.sleep(30)

# IP_ADDR = "10.42.0.1"
IP_ADDR = "192.168.3.20"
# IP_ADDR = "172.16.1.1"
PORT = 9001
ws_server = Websocket_Server(IP_ADDR, PORT)

ws_server.run()
