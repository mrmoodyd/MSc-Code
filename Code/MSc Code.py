#David Moody
#MSc Code

import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import animation
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
    mean = np.dot(x, ydata)
    return np.sum(np.dot((mean-x)**n,ydata))

def evolutionAnimation(saveFig = False):
    """
    Displays an animated plot showing the evolution of the calculated distribution over time.
    """
    fig, ax = plt.subplots()
    text = ax.text(0.02, 0.84, '\n'.join(('','','')), transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    line, = ax.plot(x, Wdata_3[0], color='k')
    ax.set_xlim(1-1.3e-12,1+1.3e-12)
    ax.set_ylim(0,0.01)
    ax.set_title('Earth-Moon System ($D^{(1)} = 0, D^{(2)} = $Constant)')
    ax.set_ylabel('Probability')
    ax.set_xlabel('Period / $\~P$')

    def animate(i, Wdata):
        """
        Defines the animation function used by 'FuncAnimation'.
        """
        text.set_text('\n'.join(('Time = {:.1f} $P_0$ (Orbits)'.format(i*dt,2),'Probability Loss = {:.2e}'.format(abs(sum(Wdata_3[0]) - sum(Wdata_3[i]))),'Standard Deviation = {:.2e} $P_0$'.format(np.sqrt(findMoment(2,Wdata_3[i]))))))
        line.set_data(x, Wdata[i])
        return line, text, 

    #Runs the animation.
    ani = animation.FuncAnimation(fig, animate, fargs = [Wdata_3], frames=T, interval=1, blit = True, repeat=False)

    #Saves animation if required.
    if saveFig:
        ani.save('Plots/Earth-Moon (D1=0).mp4', writer = animation.FFMpegWriter(fps = 30),dpi=500)
    
    plt.show()

#Defines the finite differencing size for the space (dx) and time (dt) dimensions.
dx = 1e-14
dt = 1

#Defines the x-axis to be plotted against.
x = np.arange(1-5e-12,1+5e-12,dx)

#Defines the limits for the space (J) and time (T) range being considered.
J = len(x)
T = 5000

#Define P0, H0, omega, and alpha.
P0 = 28*60*60*24
H0 = 2.27e-18
omega = 1e-5

alpha = dt/(2*dx**2)

#Defines the drift coefficient matrix (D1) using previously defined variables.
D1Values = [(3/160)*P**2*(H0**2)*(P0**2)*(-79*omega + 288*omega - 27*omega) for P in x]
D1_0 = 0*diags([D1Values[:J-1], D1Values[1:J]], [-1,1], shape=(J,J))
D1 = diags([D1Values[:J-1], D1Values[1:J]], [-1,1], shape=(J,J))

#Defines the diffusion coefficient matrix (D2) using previously defined variables.
D2_c = np.diag([(27/20)*(H0**2)*(P0**2)*omega for P in x])       #For constant D2
D2 = np.diag([(27/20)*P**3*(H0**2)*(P0**2)*omega for P in x])  #For variable D2

#Defines the initial W vector as a narrow Gaussian 
variance = (5*dx)**2
W = exactSol(variance)

#Defines the M and I matrices as outlined in the progress report.
M = diags([-1,2,-1], [-1, 0, 1], shape = (J,J)).toarray()
I = np.identity(J)

#Defines the update matrix, making use of the I and M matrices and the alpha constant.
updateMatrix_1 = np.matmul(np.linalg.inv(I + alpha*((dx/2)*D1_0 + np.matmul(M,D2_c))), I - alpha*((dx/2)*D1_0 + np.matmul(M,D2_c)))
updateMatrix_2 = np.matmul(np.linalg.inv(I + alpha*((dx/2)*D1_0 + np.matmul(M,D2))), I - alpha*((dx/2)*D1_0 + np.matmul(M,D2)))
updateMatrix_3 = np.matmul(np.linalg.inv(I + alpha*((dx/2)*D1 + np.matmul(M,D2))), I - alpha*((dx/2)*D1 + np.matmul(M,D2)))

#Set boundary Conditions.
updateMatrix_1[0] = [0]*J
updateMatrix_1[J-1] = [0]*J

updateMatrix_2[0] = [0]*J
updateMatrix_2[J-1] = [0]*J

updateMatrix_3[0] = [0]*J
updateMatrix_3[J-1] = [0]*J

#Initialise PDF data list
Wdata_1 = [W]
Wdata_2 = [W]
Wdata_3 = [W]

pLoss = [0]
momentData_1 = [findMoment(3, W)/findMoment(2,W)**(3/2)]
momentData_2 = [findMoment(3, W)/findMoment(2,W)**(3/2)]
momentData_3 = [findMoment(3, W)/findMoment(2,W)**(3/2)]

#Run simulation.
# for t in range(0,T-1):
#     """
#     Iterate over the defined number of timesteps, evolving the distribution at each timestep.
#     """
#     Wdata_1.append(np.matmul(updateMatrix_1, Wdata_1[t]).tolist()[0])     #Update the distribution at each timestep.
#     momentData_1.append(findMoment(3, Wdata_1[t+1])/findMoment(2,W)**(3/2))

#     Wdata_2.append(np.matmul(updateMatrix_2, Wdata_2[t]).tolist()[0])
#     momentData_2.append(findMoment(3, Wdata_2[t+1])/findMoment(2,W)**(3/2))

#     Wdata_3.append(np.matmul(updateMatrix_3, Wdata_3[t]).tolist()[0])
#     momentData_3.append(findMoment(3, Wdata_3[t+1])/findMoment(2,W)**(3/2))

#     variance += 2*(27/20)*(H0**2)*(P0**2)*omega*dt
#     pLoss.append(sum(W)-sum(Wdata_1[t+1]))

a=0
x = np.arange(-5,5,0.01)
W = skewnorm.pdf(x, a, loc=0, scale=0.5)/np.sum(skewnorm.pdf(x, a, loc=0, scale=0.5))
print(findMoment(3, W)/findMoment(2,W)**(3/2))
plt.plot(x,W)
plt.show()

# evolutionAnimation()
# plt.title('Standardised Third Central Moment Against Time')
# plt.xlabel('Log(Time) / Log(Orbits, $P_0$)')
# plt.ylabel('Log(Standardised Third Central Moment)')

# #plt.plot(np.log(np.linspace(0,T,T)),np.log(np.abs(pLoss)), label='Probability Loss')
# plt.plot(np.log(np.linspace(0,T,T)),np.log(np.abs(momentData_1)), label='$D^{(1)} = 0, D^{(2)} =$ Constant')
# plt.plot(np.log(np.linspace(0,T,T)),np.log(np.abs(momentData_2)), label='$D^{(1)} = 0, D^{(2)} =$ Non-Constant')
# plt.plot(np.linspace(0,T,T),np.abs(momentData_3), label='$D^{(1)} = $Non-Zero, $D^{(2)} =$ Non-Constant')

# plt.legend()
# plt.show()