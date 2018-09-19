import numpy as np
import matplotlib.pyplot as plt

v_imax = 50
v_r = 0.03

PI = np.pi
cos = np.cos
sin = np.sin

data_x = []
data_y = []
for v_i in range(0,v_imax):
    v_jmax = 5*v_i
    for v_j in range(0,v_jmax):
        #data_x.append(np.cos((2*np.pi)*i/imax)*R*(0.2))
        #data_y.append(np.sin((2*np.pi)*j/jmax)*R*(0.2))
        data_x.append( cos((2*PI)*v_j/v_jmax)*(v_i/v_imax)*v_r*(0.33) )
        data_y.append( sin((2*PI)*v_j/v_jmax)*(v_i/v_imax)*v_r*(0.33) )

print(len(data_x))
plt.scatter(data_x,data_y)
plt.show()
