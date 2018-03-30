import numpy as np
import matplotlib.pyplot as plt

from encoder import read_data


N = 2500


vals = np.zeros([N,2])

for n in range(1,N+1):
    val = read_data(n)
    vals[n-1,:] = val

plt.scatter(vals[:,0],vals[:,1])
plt.show()
