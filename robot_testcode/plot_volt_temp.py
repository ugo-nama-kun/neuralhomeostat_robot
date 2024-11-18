import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

plt.figure()

volt_hist = np.load("volt_temp_data/volt_hist.npy")
ave_volt_hist = np.load("volt_temp_data/ave_volt_hist.npy")
temp_cpu_hist = np.load("volt_temp_data/temp_cpu_hist.npy")
temp_hist = np.load("volt_temp_data/temp_hist.npy")

plt.subplot(311)
y_ = np.stack(temp_hist, axis=0).transpose()
for i in range(8):
    plt.plot(y_[i], alpha=0.5)

plt.ylim([35.0, 45.0])
plt.title("motor temp")
plt.xlabel("approx. 1 sec")
plt.ylabel("[degree]")

plt.subplot(312)
y_ = np.stack(volt_hist, axis=0).transpose()
for i in range(8):
    plt.plot(y_[i], linewidth=1, alpha=0.5)

plt.plot(ave_volt_hist, "r.-", linewidth=2)
plt.ylim([11., 12.2])
plt.title("motor voltage")
plt.xlabel("approx. 1 sec")
plt.ylabel("[V]")

plt.subplot(313)
plt.plot(temp_cpu_hist)
plt.title("CPU temp")
plt.xlabel("approx. 1 sec")
plt.ylabel("[C]")

plt.tight_layout()

plt.show()
