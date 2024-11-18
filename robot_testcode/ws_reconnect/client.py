from collections import OrderedDict

import numpy as np
import rel
import websocket  # websocket-client
import json

try:
    import thread
except ImportError:
    import _thread as thread
import time


class Websocket_Client():
    
    def __init__(self, host_addr):
        self.host_addr = host_addr
        
        # デバックログの表示/非表示設定
        websocket.enableTrace(True)
        
        # WebSocketAppクラスを生成
        # 関数登録のために、ラムダ式を使用
        self.ws = websocket.WebSocketApp(
            url=host_addr,
            on_message=lambda ws, msg: self.on_message(ws, msg),
            on_error=lambda ws, msg: self.on_error(ws, msg),
            on_close=lambda ws: self.on_close(ws),
            on_open=lambda ws: self.on_open(ws)
        )
    
    # メッセージ受信に呼ばれる関数
    def on_message(self, ws, message):
        print(message)
        
    def on_error(self, ws, msg):
        print("On Error! ", time.time())
        
    # サーバーから切断時に呼ばれる関数
    def on_close(self, ws):
        print("### closed ###")
        
    # サーバーから接続時に呼ばれる関数
    def on_open(self, ws):
        thread.start_new_thread(self.run, ())
    
    # サーバーから接続時にスレッドで起動する関数
    def run(self, *args):
        step = 1
        while True:
            print("step @ ", step)

            # if step % 20 == 0:
            #     s = "req:shutdown"
            #     time_sleep = 60
            # else:
            s = "knock knock!"
            time_sleep = 1

            print("send message: ", s)
            
            try:
                self.ws.send(s)
            except websocket.WebSocketConnectionClosedException as e:
                print("Error!: ", e)
                break

            time.sleep(time_sleep)
            step += 1
            
        print("web socket thread ended.")
            
    # websocketクライアント起動
    def run_forever(self):
        self.ws.run_forever()

ADDR = "172.16.1.1"
# ADDR = "192.168.3.20"

HOST_ADDR = f"ws://{ADDR}:9001/"
ws_client = Websocket_Client(HOST_ADDR)

while True:
    try:
        ws_client.ws.run_forever(dispatcher=rel, reconnect=3)
        rel.signal(2, rel.abort)  # Keyboard Interrupt
        rel.dispatch()
    except ConnectionResetError as e:
        ws_client = Websocket_Client(HOST_ADDR)
        print(e)
