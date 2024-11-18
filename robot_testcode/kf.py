from pykalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt

dt = 1 / 15.

tm = [[1, dt, 0.5*dt**2, 0, 0, 0, 0, 0, 0],  # pos
      [0, 1, dt, 0, 0, 0, 0, 0, 0],  # vel
      [0, 0, 1, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 1, dt, 0.5 * dt ** 2, 0, 0, 0],  # pos
      [0, 0, 0, 0, 1, dt, 0, 0, 0],  # vel
      [0, 0, 0, 0, 0, 1, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 1, dt, 0.5 * dt ** 2],  # pos
      [0, 0, 0, 0, 0, 0, 0, 1, dt],  # vel
      [0, 0, 0, 0, 0, 0, 0, 0, 1]
      ]

om = [
    [0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1]
]
kf = KalmanFilter(
    transition_matrices=tm,  # acc
    observation_matrices=om,
)

# measurements = np.sin(2 * np.pi * np.arange(100) / 50) + 0.1 * np.random.randn(100)
measurements = 1 * np.random.randn(100, 3)

# online update
mean_data = np.zeros((100, 9))
filtered_state_mean = np.zeros(9)
filtered_state_covariance = np.eye(9)
for t in range(100):
    # if t == 60:
    #     filtered_state_mean = np.zeros(9)
    #     filtered_state_covariance = np.eye(9)
    print(measurements[t])
    filtered_state_mean, filtered_state_covariance = kf.filter_update(filtered_state_mean, filtered_state_covariance, measurements[t])
    mean_data[t] = filtered_state_mean

plt.plot(measurements)
plt.plot(mean_data)
plt.legend(["obs", "pos", "vel", "acc"])
plt.show()
