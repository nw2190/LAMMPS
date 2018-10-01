import numpy as np

#c1 = [[0.001139, -0.003129], [0.000692, -0.003257], [0.000232, -0.003322]]
c1 = [[0.001471, 0.002675], [0.001603, 0.002598], [0.001732, 0.002514]]

#c2 = [[-0.010100, 0.002305], [-0.009987, 0.002756], [-0.009853, 0.003201]]
c2 = [[-0.010236, 0.002842], [-0.010193, 0.002989], [-0.010275, 0.002694]]

print(c1[0])

def l2_dist(x,y):
    return np.sqrt(np.power(x[0]-y[0],2)+np.power(x[1]-y[1],2))

d1 = []
for i in range(0,3):
    for j in range(0,3):
        d1.append(l2_dist(c1[i],c1[j]))
d1_max = np.max(d1)

d2 = []
for i in range(0,3):
    for j in range(0,3):
        d2.append(l2_dist(c2[i],c2[j]))
d2_max = np.max(d2)

print(d1_max)
print(d2_max)
