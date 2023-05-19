#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import csv
import math
from scipy.stats import norm, skewnorm
from scipy.sparse import diags

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = ""):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = f' (a = {printEnd})\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

def exactSol(variance, meanLoc=0):
    """
    Creates an exact Guassian PDF using a given variance.
    """
    scale = 1/np.sum(norm.pdf(x, loc=np.mean(x), scale=np.sqrt(variance)))
    return scale*norm.pdf(x, loc=np.mean(x), scale=np.sqrt(variance))

driftCoeff = []
diffCoeff = []

#Read binary data and D1,D2 coefficients
with open('Data/PSR-B1259-63.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(spamreader)        #Skip first line in csv
    for n,row in enumerate(spamreader):
        if n == 0:          #Load binary characteristics and name
            name = row[0]
            P0 = float(row[1])
            m1 = float(row[2])
            m2 = float(row[3])
            e = float(row[4])
        elif n>1:           #Append D1 and D2 coefficients, multiplied by n
            driftCoeff.append((n-1)*float(row[0]))
            diffCoeff.append((n-1)*float(row[1]))

    driftCoeff = np.array(driftCoeff)
    diffCoeff = np.array(diffCoeff)
    
dx = 8e-13
dt = 1000
a = dt/(2*dx**2)

x = np.arange(-15e-10, 15e-10, dx)
nLat = len(x)

#Define universal constants
H0 = 2.27e-18
G = 6.67e-11
c = 2.998e8

#Define additional parameters
am = np.sqrt(1-e**2)
massRatio = m1*m2/((m1+m2)**2)

B = (192*math.pi*massRatio*((2*math.pi*G*(m1+m2)/(c**3))**(5/3))/(5*am**7))*(1 + (73/24)*e**2 + (37/96)*e**4)**(5/3)
tc = 3*(P0**(8/3))/(8*B)

M = np.zeros((nLat, nLat))
M += 2. * np.eye(nLat) - np.eye(nLat, k=-1) - np.eye(nLat, k=+1)
M[0] = 0.
M[-1] = 0.

nt = 1000

xMat = np.outer(np.ones(nt), x)

w = np.zeros(nLat)
w[nLat//2] = 0.5
w[nLat//2 - 1] = 0.5
# variance = (5*dx)**2
# w = exactSol(variance)
wMat = np.zeros((nt, nLat))
wMat[0] = w

#Define gravitational wave intensity
peakOmega = 1e-7
omegaValue = lambda P : [peakOmega*(((n*P0)/(2*P))**3)*(7/(4+(3*((n*P0)/(2*P))**2)))**3.5 for n in range(1,401)]

for t in range(0,nt-1):
    detP = ((8*B/3)*(tc-dt*(t)*P0))**(3/8)      #Value of P in absence of GW
    dPVP = 5/(8*(tc-dt*(t)*P0))             #Derivative of VP wrt P, evaluated at detP

    print(detP)
    omega = np.array([omegaValue(detP*(1+P)) for P in x])
    D1Coeffs = omega.dot(driftCoeff)
    D2Coeffs = omega.dot(diffCoeff)

    D1Values = np.array([P*dPVP + (9/80)*(H0**2)*((1+P)**2)*(detP**2)*(am**2)*D1Coeffs[i] for i,P in enumerate(x)])
    D1 = diags([-D1Values[:nLat-1], D1Values[1:nLat]], [-1,1], shape=(nLat,nLat)).toarray()
 
    D2 = np.diag([(27/20)*((1+P)**3)*(detP**2)*(H0**2)*(am**2)*D2Coeffs[i] for i,P in enumerate(x)])
    
    MA = np.eye(nLat) + a*(dx/2)*D1 + a*M.dot(D2)
    MB = np.eye(nLat) - a*(dx/2)*D1 - a*M.dot(D2)
    U = np.linalg.inv(MA).dot(MB)

    w = U.dot(w)
    wMat[t+1] = w
    printProgressBar(t+1, nt-1, prefix = 'Progress:', suffix = 'Complete', length = 50, printEnd = a*np.mean(np.diagonal(D2)))

mu = np.sum(xMat * wMat, axis=-1)
muMat = np.outer(mu, np.ones(nLat))
sigma = np.sum((xMat - muMat)**2. * wMat, axis=-1) ** 0.5
skew = np.sum((xMat - muMat)**3. * wMat, axis=-1)
kurt = np.sum((xMat - muMat)**4. * wMat, axis=-1)
exKurt = (kurt / 3. / sigma**4.) - 1.

kurtList = []
for kurtVal in exKurt[1:]:
    kurtList.append(kurtVal)
plt.plot(range(1,nt), kurtList)
plt.show()