#David Moody
#MSc Code

import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import animation
plt.rcParams['animation.ffmpeg_path'] = r'C:\FFmpeg\bin\ffmpeg'
from scipy.stats import norm
from scipy.sparse import diags
from scipy.signal import peak_widths, find_peaks

def exactSol(variance):
    """
    Creates an exact Guassian PDF using a given variance.
    """
    return norm.pdf(x, loc=1, scale=np.sqrt(variance))/np.sum(norm.pdf(x, loc=1, scale=np.sqrt(variance)))

def findMoment(n,ydata):
    """
    Takes the y data of the pdf and calculates the central moment specified by n.
    """
    mean = np.dot(x, ydata)
    return np.sum(np.dot((mean-x)**n,ydata))

def evolutionAnimation(saveFig = False):
    """
    Displays an animated plot showing the evolution of the calculated distribution over time.
    """
    fig, ax = plt.subplots()
    text = ax.text(0.02, 0.84, '\n'.join(('','','')), transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    line, = ax.plot(x, Wdata[0], color='k')
    ax.set_xlim(1-1.3e-12,1+1.3e-12)
    ax.set_ylim(0,0.1)
    ax.set_title('Earth-Moon System ($D^{(1)} = 0, D^{(2)} = $Constant)')
    ax.set_ylabel('Probability')
    ax.set_xlabel('Period / $\~P$')

    def animate(i, Wdata):
        """
        Defines the animation function used by 'FuncAnimation'.
        """
        text.set_text('\n'.join(('Time = {:.1f} $P_0$ (Orbits)'.format(i*dt,2),'Probability Loss = {:.2e}'.format(abs(sum(Wdata[0]) - sum(Wdata[i]))),'Standard Deviation = {:.2e} $P_0$'.format(np.sqrt(findMoment(2,Wdata[i]))))))
        line.set_data(x, Wdata[i])
        return line, text, 

    #Runs the animation.
    ani = animation.FuncAnimation(fig, animate, fargs = [Wdata], frames=T, interval=1, blit = True, repeat=False)

    #Saves animation if required.
    if saveFig:
        ani.save('Plots/Earth-Moon (D1=0).mp4', writer = animation.FFMpegWriter(fps = 30),dpi=500)
    
    plt.show()

#Defines the finite differencing size for the space (dx) and time (dt) dimensions.
dx = 1e-14
dt = 0.5

#Defines the x-axis to be plotted against.
x = np.arange(1-5e-12,1+5e-12,dx)

#Defines the limits for the space (J) and time (T) range being considered.
J = len(x)
T = 401

#Define P0, H0, omega, and alpha.
P0 = 28*60*60*24
H0 = 2.27e-18
omega = 1e-5

alpha = dt/(2*dx**2)

#Defines the drift coefficient matrix (D1) using previously defined variables.
D1Values = [(3/160)*P**2*(H0**2)*(P0**2)*(-79*omega + 288*omega - 27*omega) for P in x]
D1 = 0*diags([D1Values[:J-1], D1Values[1:J]], [-1,1], shape=(J,J))

#Defines the diffusion coefficient matrix (D2) using previously defined variables.
D2 = np.diag([(27/20)*(H0**2)*(P0**2)*omega for P in x])       #For constant D2
#D2 = np.diag([(27/20)*P**3*(H0**2)*(P0**2)*omega for P in x])  #For variable D2

#Defines the initial W vector as a narrow Gaussian 
variance = dx**2
W = exactSol(variance)

#Defines the M and I matrices as outlined in the progress report.
M = diags([-1,2,-1], [-1, 0, 1], shape = (J,J)).toarray()
I = np.identity(J)

#Defines the update matrix, making use of the I and M matrices and the alpha constant.
updateMatrix = np.matmul(np.linalg.inv(I + alpha*((dx/2)*D1 + np.matmul(M,D2))), I - alpha*((dx/2)*D1 + np.matmul(M,D2)))

#Set boundary Conditions.
updateMatrix[0] = [0]*J
updateMatrix[J-1] = [0]*J

#Initialise PDF data list
Wdata = [W]

#Run simulation.
for t in range(0,T-1):
    """
    Iterate over the defined number of timesteps, evolving the distribution at each timestep.
    """

    W = np.matmul(updateMatrix, W).tolist()[0]         #Update the distribution at each timestep.
    Wdata.append(W)