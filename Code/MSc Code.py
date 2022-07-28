#David Moody
#MSc Code

import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.lines import Line2D
plt.rcParams['animation.ffmpeg_path'] = r'C:\FFmpeg\bin\ffmpeg'
from scipy.stats import norm, skewnorm
from scipy.sparse import diags

def exactSol(variance):
    """
    Creates an exact Guassian PDF using a given variance.
    """
    return norm.pdf(x, loc=1, scale=np.sqrt(variance))/np.sum(norm.pdf(x, loc=1, scale=np.sqrt(variance)))

def findMoment(n,ydata):
    """
    Takes the y data of the pdf and calculates the central moment specified by n.
    """
    mean = np.dot(x, ydata/np.sum(ydata))
    return np.dot((x-mean)**n,ydata)

def evolutionAnimation(plotData, saveFig = False):
    """
    Displays an animated plot showing the evolution of the calculated distribution over time.
    """
    fig, ax = plt.subplots()

    text = ax.text(0.02, 0.84, '\n'.join(('','','')), transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    line, = ax.plot(x, plotData[0], color='k', label='Plotted Data')
    #line1, = ax.plot(x, exactData[0], label = 'Exact Data')
    meanLine = Line2D([0], [0], color='red', label='Mean')

    handles, labels = plt.gca().get_legend_handles_labels()
    handles.extend([meanLine])
    legend = ax.legend(handles=handles)

    ax.set_ylim(0,0.01)
    ax.set_title('Earth-Moon System')
    ax.set_ylabel('Probability')
    ax.set_xlabel('Period / $\~P$')

    def animate(i, plotData, exactData):
        """
        Defines the animation function used by 'FuncAnimation'.
        """
        text.set_text('\n'.join(('Time = {:.1f} $P_0$ (Orbits)'.format(i*dt,2),'Probability Loss = {:.2e}'.format(abs(sum(plotData[0]) - sum(plotData[i]))),'Mean Drift = {:.2e} $P_0$'.format(np.dot(x,plotData[i]/np.sum(plotData[i]))-np.dot(x,exactData[i]/np.sum(exactData[i]))))))
        line.set_data(x, plotData[i])
        meanLine = ax.axvline(np.dot(x,plotData[i]/np.sum(plotData[i])), color='red', label='Mean')
        return line, text, meanLine

    #Runs the animation.
    ani = animation.FuncAnimation(fig, animate, fargs = [plotData, exactData], frames=T, interval=1, blit = True, repeat=False)

    #Saves animation if required.
    if saveFig:
        ani.save('Plots/Earth-Moon (D1=0).mp4', writer = animation.FFMpegWriter(fps = 30),dpi=500)
    
    plt.show()

#Defines the finite differencing size for the space (dx) and time (dt) dimensions.
dx = 1e-14
dt = 10

#Defines the x-axis to be plotted against.
x = np.arange(1-10e-12,1+10e-12,dx)

#Defines the limits for the space (J) and time (T) range being considered.
J = len(x)
T = 1000

#Define P0, H0, omega, and alpha.
P0 = 28*60*60*24
H0 = 2.27e-18
omega = 1e-5

alpha = dt/(2*dx**2)

#Defines the drift coefficient matrix (D1) using previously defined variables.
D1Values = np.array([(3/160)*P**2*(H0**2)*(P0**2)*(-79*omega + 288*omega - 27*omega) for P in x])
D1_0 = np.zeros((J,J))
D1 = diags([D1Values[:J-1], -D1Values[1:J]], [-1,1], shape=(J,J)).toarray()

#Defines the diffusion coefficient matrix (D2) using previously defined variables.
D2_c = np.diag([(27/20)*(H0**2)*(P0**2)*omega for P in x])       #For constant D2
D2 = np.diag([(27/20)*P**3*(H0**2)*(P0**2)*omega for P in x])  #For variable D2

#Defines the initial W vector as a narrow Gaussian 
variance = (20*dx)**2
W = exactSol(variance)

#Defines the M and I matrices as outlined in the progress report.
M = diags([-1,2,-1], [-1, 0, 1], shape = (J,J)).toarray()
I = np.identity(J)

#Defines the update matrix, making use of the I and M matrices and the alpha constant.
updateMatrix_1 = np.matmul(np.linalg.inv(I + alpha*((dx/2)*D1_0 + np.matmul(M,D2_c))), I - alpha*((dx/2)*D1_0 + np.matmul(M,D2_c)))
updateMatrix_2 = np.matmul(np.linalg.inv(I + alpha*((dx/2)*D1_0 + np.matmul(M,D2))), I - alpha*((dx/2)*D1_0 + np.matmul(M,D2)))
updateMatrix_3 = np.matmul(np.linalg.inv(I + alpha*((dx/2)*D1 + np.matmul(M,D2))), I - alpha*((dx/2)*D1 + np.matmul(M,D2)))

#Set boundary Conditions.
updateMatrix_1[0,:] = updateMatrix_1[J-1,:] = 0

updateMatrix_2[0,:] = updateMatrix_2[J-1,:] = 0

updateMatrix_3[0,:] = updateMatrix_3[J-1,:] = 0

#Initialise PDF data list
exactData = [W]
Wdata_1 = [W]
Wdata_2 = [W]
Wdata_3 = [W]

exactMoment = [findMoment(3, W)/findMoment(2, W)**(3/2)]
momentData_1 = [findMoment(3, W)/findMoment(2, W)**(3/2)]
momentData_2 = [findMoment(3, W)/findMoment(2, W)**(3/2)]
momentData_3 = [findMoment(3, W)/findMoment(2, W)**(3/2)]

#Run simulation.
for t in range(0,T-1):
    """
    Iterate over the defined number of timesteps, evolving the distribution at each timestep.
    """
    variance += 2*(27/20)*(H0**2)*(P0**2)*omega*dt
    exactData.append(exactSol(variance))
    exactMoment.append(findMoment(3, exactData[t+1])/findMoment(2, exactData[t+1])**(3/2))

    Wdata_1.append(np.matmul(updateMatrix_1, Wdata_1[t]))
    momentData_1.append(findMoment(3, Wdata_1[t+1])/findMoment(2, Wdata_1[t+1])**(3/2))

    Wdata_2.append(np.matmul(updateMatrix_2, Wdata_2[t]))
    momentData_2.append(findMoment(3, Wdata_2[t+1])/findMoment(2, Wdata_2[t+1])**(3/2))

    Wdata_3.append(np.matmul(updateMatrix_3, Wdata_3[t]))
    momentData_3.append(findMoment(3, Wdata_3[t+1])/findMoment(2, Wdata_3[t+1])**(3/2))

# evolutionAnimation(exactData)
plt.title('Skewness Against Time')
plt.ylabel('$\mu_3 / \sigma^3$')
plt.xlabel('Time / $P_0$')

# plt.plot(np.log10(np.linspace(0,dt*T,T)),np.log10(exactMoment), label='Exact')
# plt.plot(np.log10(np.linspace(0,dt*T,T)),np.log10(momentData_1), label='$D^{(1)} = 0, D^{(2)} =$ Constant')
# plt.plot(np.log10(np.linspace(0,dt*T,T)),np.log10(momentData_2), label='$D^{(1)} = 0, D^{(2)} =$ Non-Constant')
# plt.plot(np.log10(np.linspace(0,dt*T,T)),np.log10(momentData_3), label='$D^{(1)} = $Non-Zero, $D^{(2)} =$ Non-Constant')

plt.plot(np.linspace(0,dt*T,T),exactMoment, label='Exact')
plt.plot(np.linspace(0,dt*T,T),momentData_1, label='$D^{(1)} = 0, D^{(2)} =$ Constant')
plt.plot(np.linspace(0,dt*T,T),momentData_2, label='$D^{(1)} = 0, D^{(2)} =$ Non-Constant')
plt.plot(np.linspace(0,dt*T,T),momentData_3, label='$D^{(1)} = $Non-Zero, $D^{(2)} =$ Non-Constant')

plt.legend()
plt.show()