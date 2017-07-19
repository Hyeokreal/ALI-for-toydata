import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

a = [[1, 1]] * 100
# map(lambda x: np.random.multivariate_normal([x[0], x[1]], [[0.0, 0.01],
#                                                               [0.01, 0.0]]), a)

# x,y = np.random.multivariate_normal([1, 1], [[0.0, 0.1], [0.1, 0.0]],size=1000).T

x = []
y = []

colors = [(0/255, 0/255, 255/255),(0/255, 40/255, 255/255),(0/255, 80/255, 255/255),(0/255, 120/255, 255/255),(0/255, 160/255, 255/255),

          ]

for x_ in range(-4, 4 + 1, 2):
    for y_ in range(-4, 4 + 1, 2):
        i, j = np.random.multivariate_normal([x_, y_], [[0.001, 0.],
                                                        [0., 0.001]], size=100).T



        print(x_,y_)
        x.append(i)
        y.append(j)

fig, ax = plt.subplots()
# plt.figure()
rgb = np.random.random((2500, 3))

x = np.asarray(x)
y = np.asarray(y)

print(x.shape)
print(y.shape)

# print(x.shape)

ax.scatter(x, y, marker='.', facecolor=((0/255, 51/255, 204/255)))

#
# c = np.asarray(b)
#
#

plt.show()
