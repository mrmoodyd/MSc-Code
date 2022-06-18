#David Moody
#MSc Code

import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import animation
plt.rcParams['animation.ffmpeg_path'] = r'C:\FFmpeg\bin\ffmpeg'
from scipy.sparse import diags
from scipy.signal import unit_impulse

#Defines the limits for the space (J) and time (T) range being considered.
J = 101
T = 2000

#Defines the finite differencing size for the space (dx) and time (dt) dimensions.
dx = 0.01
dt = 0.001

#Defines the diffusion coefficient (D2) and the constant alpha using previously defined variables.
D2 = 0.01
alpha = D2*dt/(2*dx**2)

#Defines the initial W vector as a Dirac-delta function 
W = unit_impulse(J, 'mid')

#Defines the M and I matrices as outlined in the progress report.
M = diags([-1,2,-1], [-1, 0, 1], shape = (J,J))
I = np.identity(J)

#Defines the update matrix, making use of the I and M matrices and the alpha constant.
updateMatrix = np.matmul(np.linalg.inv(I + alpha*M), I - alpha*M)

#Defines the x-axis to be plotted against.
x = np.linspace(0.5,1.5,J)

def exactSol(x,t):
    """
    Creates an exact Guassian solution for an input (x) at a given time (t), 
    using a given diffusion coefficient (D2).
    """
    sigma = 2*D2*t          #Calculates the current standard deviation
    return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-x**2 / (2 * sigma**2))  #Returns the Guassian distribution


def exactSolPlot():
    """
    Displays a log plot of the difference between the exact solution and the calculated distribution
    as a function of time.
    """
    plt.title('Exact Solution Error Against Time')
    plt.xlabel('log(t)')
    plt.ylabel('log(max(error))')
    plt.plot(np.log(np.linspace(1,T,T-1)),np.log(EData), color='k')
    plt.savefig('Plots/Exact_Error.png')

def evolutionAnimation(saveFig = False):
    """
    Displays an animated plot showing the evolution of the calculated distribution over time.
    """
    fig, ax = plt.subplots()
    fig.set_size_inches(6,4)
    text = ax.text(0.02, 0.88, '\n'.join(('','')), transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    line, = ax.plot(x, Wdata[0], color='k')
    line.axes.axis([0.5, 1.5, 0, 1])

    def animate(i, Wdata):
        """
        Defines the animation function used by 'FuncAnimation'.
        """
        ax.set_title('FPE Evolution with Time ($D^{(1)} = 0, D^{(2)} =$ Constant)')
        ax.set_ylabel('Probability')
        ax.set_xlabel('Period / $\~P$')
        text.set_text('\n'.join(('Timestep = '+str(i),'Total Probability = '+str(round(sum(Wdata[i]),2)))))
        line.set_data(x, Wdata[i])
        return line, text,

    #Runs the animation.
    ani = animation.FuncAnimation(fig, animate, fargs = [Wdata], frames=T, interval=10, blit = True, repeat=False)

    #Saves animation if required.
    if saveFig:
        ani.save('Plots/diffusion_animation.mp4', writer = animation.FFMpegWriter(fps = 30),dpi=500)
    
    plt.show()

EData = []
Wdata = []

for t in range(T):
    """
    Iterate over the defined number of timesteps, evolving the distribution at each timestep 
    and calculating any other wanted parameters (errors).
    """
    if t > 0:
        EData.append(max(exactSol(x,t) - W))        #Calculates the max difference from the exact solution at this timestep.
    
    Wdata.append(W)
    W = np.dot(updateMatrix, W).tolist()[0]         #Update the distribution at each timestep.