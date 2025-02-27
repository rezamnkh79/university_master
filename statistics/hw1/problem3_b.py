import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-2, 2, 500)
y = (np.where((x >= -1) & (x <= 0), 1, 0) +
     np.where((x >= 0) & (x <= 1), 1, 0))

plt.plot(x, y)
mean = np.mean(x * y)
median = np.median(x[y > 0])
plt.axvline(x=float(mean), color='r', linestyle='--', label='Mean')
plt.axvline(x=float(median), color='b', linestyle='--', label='Median')
plt.axvline(x=-0.9, color='g', linestyle='--', label='Mode 1')
plt.axvline(x=0.9, color='g', linestyle='--', label='Mode 2')
plt.legend()
plt.show()
