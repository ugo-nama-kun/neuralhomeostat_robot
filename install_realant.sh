# Installing Battery monitor in the raspberry pi

# Python
sudo pip install git+https://github.com/Pithikos/python-websocket-server
sudo pip install ipget smbus2

# Service setup
sudo cp setup_files/start_realant.service /etc/systemd/system/
sudo systemctl enable start_realant.service
sudo systemctl daemon-reload
sudo systemctl restart start_realant.service

# sudo reboot now
echo "Realant installed."
