# Licensed under MIT licence, see LICENSE for details.
# Copyright Ote Robotics Ltd. 2020

import zmq
import time

ctx = zmq.Context()
sock = ctx.socket(zmq.PUB)
sock.connect("tcp://127.0.0.1:3002")

time.sleep(0.1)
sock.send_multipart([b"cmd", b"attach_servos\n"])

try:
    while True:
        time.sleep(0.1)
        sock.send_multipart([b"cmd", b"s1 512 s2 512 s3 512 s4 512 s5 512 s6 512 s7 512 s8 512\n"])
        time.sleep(1)
        sock.send_multipart([b"cmd", b"s1 512 s2 224 s3 512 s4 224 s5 512 s6 224 s7 512 s8 224\n"])
        time.sleep(1)
finally:
    sock.send_multipart([b"cmd", b"reset\n"])
    time.sleep(1)
    sock.send_multipart([b"cmd", b"detach_servos\n"])

    sock.close()
    ctx.term()
