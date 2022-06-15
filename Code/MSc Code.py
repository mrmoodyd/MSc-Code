#David Moody
#MSc Code

import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse import diags
from scipy.signal import unit_impulse

J = 101
dt = 0.001
dx = 0.01
D = 0.001

x = np.linspace(-1,1,J)
W = unit_impulse(J, 'mid')
M = diags([-1,2,-1], [-1, 0, 1], shape = (J,J))
I = np.identity(J)

alpha = D*dt/(2*dx**2)
T = 2000

updateMatrix = np.matmul(np.linalg.inv(I + alpha*M), I - alpha*M)

Wdata = []
for t in range(T):
    if t % 10 == 0:
        Wdata.append(W)
    W = np.dot(updateMatrix, W).tolist()[0]

fig, ax = plt.subplots()
line, = ax.plot(x, Wdata[0], color='k')

def animate(i, Wdata):
    ax.clear()
    line.set_data(x, Wdata[i])
    line.axes.axis([-1, 1, 0, 1])
    return line,

ani = FuncAnimation(fig, animate, fargs = [Wdata], frames=T, interval=0.01, blit=True, repeat=False)

plt.show()