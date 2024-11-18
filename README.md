# Neural Homeostat Experiment Code (Robot & Simulation)

## Simulation
```bash
# (create a virtual environment if necessary)

# install simulator
pip install ./homeostatic_robot_sim_env
# Running simulation. IS_SIM=True in the code (Line 19).
python run_experiment.py
```
## Hardware 
An extension of RealAnt for the autonomous learning

realant from: https://github.com/OteRobotics/realant, https://shop.oterobotics.com/products/realant-fully-assembled

- servo motors AX-12A: https://emanual.robotis.com/docs/en/dxl/ax/ax-12a/
- how to get voltage of the motor: https://emanual.robotis.com/docs/en/dxl/ax/ax-12a/#present-voltage
- how to get motor temperatures: https://emanual.robotis.com/docs/en/dxl/ax/ax-12a/#present-temperature-43
## Software
### install (battery-monitor/raspberry-pi side)

1. enable ssh by `sudo raspy-config` -> Interface Options -> SSH 
2. set stati IP address (192.168.1.101 by default in this app)
3. run installation script below:

```bash
cd ~
git clone git@github.com:ugo-nama-kun/neuralhomeostat_robot.git
cd neuralhomeostat_robot
sh install_realant.sh
```

### package requirements of python (Control-Experiment PC side)

```bash
# install necessary packages 
pip install gymnasium>=0.27.1 numpy<=1.23.3 rel websocket-client==1.4.1 ping3 pyserial pyzmq hooman

# install optitrack package
# Optitrack broadcast data to 192.168.1.36 and it is "Y-up" setting.
cd ~
git clone https://github.com/mje-nz/python_natnet
cd python_natnet
pip install -e .
cd -
```

```bash
# first. run server. Configure network params in network.py
python start_robot_manager.py
```

```bash
# test run
python test_system.py
```

```bash
# walking test
python test_walk.py
```

## OptiTrack configurations

1. Make two rigid bodies: ANT (on the top of the robot) and FOOD (on the top of the target object)
2. Streaming ID should be ANT = 1, FOOD = 2 (can be setup in Motive RigidBody setting)

### tips

#### Monitoring robot temperatures

```bash
python start_robot_manager.py
# another shell window
python start_robot_monitoring.py
```


#### zmq returns errors.
kill `python env_server.py` if thread is alive. look for pid by `ps ax | grep python`.

#### enable/disable the battery monitoring script in rasberry-pi after the installation

```bash
# enable
sudo cp setup_files/start_realant.service /etc/systemd/system 
sudo systemctl enable start_realant.service
```

```bash
# disable
sudo systemctl disable start_realant.service
```

#### my raspberry-pi settings
```
raspi settings:

username: realant
```

display tips: https://qiita.com/karaage0703/items/97808dfb957b3312b649

# When you facing Errors...
### Raspi4 is not working
```bash
- Failed to get present position 0xFF
- Failed to get present temperature 0x80
```
