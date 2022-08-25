#David Moody
#MSc Code

import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.lines import Line2D
plt.rcParams['animation.ffmpeg_path'] = r'C:\FFmpeg\bin\ffmpeg'
from scipy.stats import norm, skewnorm
from scipy.sparse import diags
import csv

def exactSol(variance, meanLoc=1):
    """
    Creates an exact Guassian PDF using a given variance.
    """
    scale = 1/np.sum(norm.pdf(x, loc=meanLoc, scale=np.sqrt(variance)))
    return scale*norm.pdf(x, loc=meanLoc, scale=np.sqrt(variance))

def findMoment(n,ydata):
    """
    Takes the y data of the pdf and calculates the central moment specified by n.
    """
    ydata /= np.sum(ydata)                  #Normalise y data

    mean = np.dot(x, ydata)
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

    ax.set_ylim(0,0.04)
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
dx = 5e-14
dt = 500

xlim = 6.5e-10

#Defines the x-axis to be plotted against.
x = np.arange(1-xlim,1+xlim,dx)

#Defines the limits for the space (J) and time (T) range being considered.
J = len(x)
T = 26075

#Define P0, H0, omega, and alpha.
P0 = 28*60*60*24
H0 = 2.27e-18
omega = 1e-5

alpha = dt/(2*dx**2)

#Defines the drift coefficient matrix (D1) using previously defined variables.
D1Values = np.array([(3/160)*P**2*(H0**2)*(P0**2)*(288*omega) for P in x])
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
updateMatrix_1 = np.matmul(np.linalg.inv(I + alpha*(np.matmul(M,D2_c))), I - alpha*(np.matmul(M,D2_c)))
updateMatrix_2 = np.matmul(np.linalg.inv(I + alpha*(np.matmul(M,D2))), I - alpha*(np.matmul(M,D2)))
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

#Initialise 4th moment lists
exact4Moment = [findMoment(4, W)]
moment4Data_1 = [findMoment(4, W)]
moment4Data_2 = [findMoment(4, W)]
moment4Data_3 = [findMoment(4, W)]

#Initialise 3rd moment lists
exact3Moment = [findMoment(3,W)]
moment3Data_1 = [findMoment(3,W)]
moment3Data_2 = [findMoment(3,W)]
moment3Data_3 = [findMoment(3,W)]

#Initialise 2nd moment lists
exact2Moment = [findMoment(2,W)]
moment2Data_1 = [findMoment(2,W)]
moment2Data_2 = [findMoment(2,W)]
moment2Data_3 = [findMoment(2,W)]

#Run simulation.
for t in range(0,T-1):
    """
    Iterate over the defined number of timesteps, evolving the distribution at each timestep.
    Also save data regarding 1st, 2nd, and 3rd moments.
    """
    print(str(100*t/T) + '% complete.')
    variance += 2*(27/20)*(H0**2)*(P0**2)*omega*dt
    exactData.append(exactSol(variance))
    exact2Moment.append(findMoment(2, exactData[t+1]))
    exact3Moment.append(findMoment(3, exactData[t+1]))
    exact4Moment.append(findMoment(4, exactData[t+1]))

    Wdata_1.append(np.matmul(updateMatrix_1, Wdata_1[t]))
    moment2Data_1.append(findMoment(2, Wdata_1[t+1]))
    moment3Data_1.append(findMoment(3, Wdata_1[t+1]))
    moment4Data_1.append(findMoment(4, Wdata_1[t+1]))

    Wdata_2.append(np.matmul(updateMatrix_2, Wdata_2[t]))
    moment2Data_2.append(findMoment(2, Wdata_2[t+1]))
    moment3Data_2.append(findMoment(3, Wdata_2[t+1]))
    moment4Data_2.append(findMoment(4, Wdata_2[t+1]))

    Wdata_3.append(np.matmul(updateMatrix_3, Wdata_3[t]))
    moment2Data_3.append(findMoment(2, Wdata_3[t+1]))
    moment3Data_3.append(findMoment(3, Wdata_3[t+1]))
    moment4Data_3.append(findMoment(4, Wdata_3[t+1]))

#Write data to csv files
with open('Data/2Moment_T = '+str(T)+'.csv', 'w+', newline='') as csvfile:
    fileWriter = csv.writer(csvfile, delimiter=',')
    fileWriter.writerow(['#dx',dx])
    fileWriter.writerow(['#dt',dt])
    fileWriter.writerow(['#xlim',xlim])
    fileWriter.writerow(['Timestep', 'Exact 2nd Moment', '2nd Moment 1', '2nd Moment 2', '2nd Moment 3'])
    for i in range(0,T-1):
        fileWriter.writerow([dt*i, exact2Moment[i], moment2Data_1[i], moment2Data_2[i], moment2Data_3[i]])

with open('Data/3Moment_T = '+str(T)+'.csv', 'w+', newline='') as csvfile:
    fileWriter = csv.writer(csvfile, delimiter=',')
    fileWriter.writerow(['#dx',dx])
    fileWriter.writerow(['#dt',dt])
    fileWriter.writerow(['#xlim',xlim])
    fileWriter.writerow(['Timestep', 'Exact 3rd Moment', '3rd Moment 1', '3rd Moment 2', '3rd Moment 3'])
    for i in range(0,T-1):
        fileWriter.writerow([dt*i, exact3Moment[i], moment3Data_1[i], moment3Data_2[i], moment3Data_3[i]])

with open('Data/4Moment_T = '+str(T)+'.csv', 'w+', newline='') as csvfile:
    fileWriter = csv.writer(csvfile, delimiter=',')
    fileWriter.writerow(['#dx',dx])
    fileWriter.writerow(['#dt',dt])
    fileWriter.writerow(['#xlim',xlim])
    fileWriter.writerow(['Timestep', 'Exact 4th Moment', '4th Moment 1', '4th Moment 2', '4th Moment 3'])
    for i in range(0,T-1):
        fileWriter.writerow([dt*i, exact4Moment[i], moment4Data_1[i], moment4Data_2[i], moment4Data_3[i]])