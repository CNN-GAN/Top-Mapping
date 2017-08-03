import numpy as np
import matplotlib.pyplot as plt

N = 5
y1 = [20,10,30,25,15]
y2 = [15,14,34,10, 5]
index = np.arange(5)

x = np.arange(len(index))
bar_width = 0.3
plt.bar(index, y1, width=0.3, color='y')
plt.bar(index+bar_width, y2, width=0.3, color='r')
plt.xticks(x + bar_width, map(str, x))
plt.show()
