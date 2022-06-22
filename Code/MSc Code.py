#David Moody
#MSc Code

import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import animation
plt.rcParams['animation.ffmpeg_path'] = r'C:\FFmpeg\bin\ffmpeg'
from scipy.stats import norm
from scipy.sparse import diags
from scipy.signal import unit_impulse

def exactSol(variance):
    """
    Creates an exact Guassian solution for an input (x) at a given time (t), 
    using a given variance.
    """

    return dx*norm.pdf(x, loc=1, scale=np.sqrt(variance))

def evolutionAnimation(saveFig = False):
    """
    Displays an animated plot showing the evolution of the calculated distribution over time.
    """
    fig, ax = plt.subplots()
    text = ax.text(0.02, 0.87, '\n'.join(('','')), transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    line, = ax.plot(x, Wdata[0], color='k')

    def animate(i, Wdata):
        """
        Defines the animation function used by 'FuncAnimation'.
        """
        ax.set_title('FPE Evolution with Time ($D^{(1)} = 0$)')
        ax.set_ylabel('Probability')
        ax.set_xlabel('Period / $\~P$')
        text.set_text('\n'.join(('Time = {:.2f} $\~P$ (Orbits)'.format(i*dt,2),'Probability Loss = {:.2e}'.format(1 - sum(Wdata[i])))))
        line.set_data(x, Wdata[i])
        return line, text,

    #Runs the animation.
    ani = animation.FuncAnimation(fig, animate, fargs = [Wdata], frames=T, interval=10, blit = True, repeat=False)

    #Saves animation if required.
    if saveFig:
        ani.save('Plots/FPE_D1=0.mp4', writer = animation.FFMpegWriter(fps = 30),dpi=500)
    
    plt.show()

#Defines the finite differencing size for the space (dx) and time (dt) dimensions.
dx = 1e-15
dt = 0.01

#Defines the x-axis to be plotted against.
x = np.arange(1-1e-13,1+1e-13,dx)

#Defines the limits for the space (J) and time (T) range being considered.
J = len(x)
T = 2000

#Define P0 in seconds
P0 = (1/12)*3.154e7

#Defines the diffusion coefficient (D2) and the constant alpha using previously defined variables.
H0 = 2.27e-18
omega = 1e-6
D2 = np.diag([(27/20)*P*(H0**2)*(P0**2)*omega for P in x])
alpha = dt/(2*dx**2)

#Defines the initial W vector as a narrow Gaussian 
variance = dx**2
W = exactSol(variance)

#Defines the M and I matrices as outlined in the progress report.
M = diags([-1,2,-1], [-1, 0, 1], shape = (J,J)).toarray()
I = np.identity(J)

#Defines the update matrix, making use of the I and M matrices and the alpha constant.
updateMatrix = np.matmul(np.linalg.inv(I + alpha*np.dot(M,D2)), I - alpha*np.dot(M,D2))

Wdata = [W]

for t in range(0,T-1):
    """
    Iterate over the defined number of timesteps, evolving the distribution at each timestep 
    and calculating any other wanted parameters (errors).
    """

    W = np.dot(updateMatrix, W)         #Update the distribution at each timestep.
    Wdata.append(W)