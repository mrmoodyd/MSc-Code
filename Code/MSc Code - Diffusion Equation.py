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

def exactSolPlot(saveFig=False):
    """
    Displays a log plot of the difference between the exact solution and the calculated distribution
    as a function of time.
    """
    
    plt.title('Exact Solution Error Against Time')
    plt.xlabel('log(t) / log($\~P$)')
    plt.ylabel('log(max(error))')
    plt.plot(np.log(dt*np.linspace(0,T,T)),np.log(EData), color='k')

    if saveFig:
        plt.savefig('Plots/Exact_Error.png')
        
    plt.show()

def evolutionAnimation(saveFig = False):
    """
    Displays an animated plot showing the evolution of the calculated distribution over time.
    """
    fig, ax = plt.subplots()
    text = ax.text(0.02, 0.85, '\n'.join(('','','')), transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    line, = ax.plot(x, Wdata[0], color='k')

    def animate(i, Wdata, EData):
        """
        Defines the animation function used by 'FuncAnimation'.
        """
        ax.set_title('FPE Evolution with Time ($D^{(1)} = 0, D^{(2)} =$ Constant)')
        ax.set_ylabel('Probability')
        ax.set_xlabel('Period / $\~P$')
        text.set_text('\n'.join(('Time = {:.2f} $\~P$'.format(i*dt,2),'Probability Loss = {:.2e}'.format(1-sum(Wdata[i])),'Exact Solution Error = {:.2e}'.format(EData[i]))))
        line.set_data(x, Wdata[i])
        return line, text,

    #Runs the animation.
    ani = animation.FuncAnimation(fig, animate, fargs = [Wdata, EData], frames=T, interval=1, blit = True, repeat=False)

    #Saves animation if required.
    if saveFig:
        ani.save('Plots/diffusion_animation.mp4', writer = animation.FFMpegWriter(fps = 30),dpi=500)
    
    plt.show()


#Defines the finite differencing size for the space (dx) and time (dt) dimensions.
dx = 0.001
dt = 0.001

#Defines the x-axis to be plotted against.
xmin = 0.5
xmax = 1.5
x = np.arange(xmin,xmax,dx)

#Defines the limits for the space (J) and time (T) range being considered.
J = len(x)
T = 2000

#Defines the diffusion coefficient (D2) and the constant alpha using previously defined variables.
D2 = 0.01
alpha = D2*dt/(2*dx**2)

#Defines the initial W vector as a narrow Gaussian 
variance = 10*dx**2
W = exactSol(variance)

#Defines the M and I matrices as outlined in the progress report.
M = diags([-1,2,-1], [-1, 0, 1], shape = (J,J))
I = np.identity(J)

#Defines the update matrix, making use of the I and M matrices and the alpha constant.
updateMatrix = np.matmul(np.linalg.inv(I + alpha*M), I - alpha*M)

#Implement boundary conditions
updateMatrix[0] = [0]*J
updateMatrix[J-1] = [0]*J

EData = [0]
exactData = [W]
Wdata = [W]

for t in range(0,T-1):
    """
    Iterate over the defined number of timesteps, evolving the distribution at each timestep 
    and calculating any other wanted parameters (errors).
    """
    variance += 2*D2*dt
    E = exactSol(variance)
    exactData.append(E)

    W = np.dot(updateMatrix, W).tolist()[0]         #Update the distribution at each timestep.
    Wdata.append(W)

    EData.append(np.max(E - W))        #Calculates the max difference from the exact solution at this timestep.
