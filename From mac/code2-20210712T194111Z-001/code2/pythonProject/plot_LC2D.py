""" 
Script to plot the Lissajous-Chebyshev node points LC and 
the corresponding spectral index sets
(C) P. Dencker and W. Erb 01.10.2016

This python implementation was written by W. Erb (01.10.2018)
"""

import LC2Ditp as LC
from math import gcd
import numpy as np
import matplotlib.pyplot as plt

#Set the parameters (please change here)
LCrange = [-1, 1, -1, 1]         #Range of area covered by Lissajous curves
m       = [17,16]                #Frequency parameters (positive integers)
kappa   = [0,0]                  #Phase shift parameter (positive integers)

#Generate LC points and prepare plot for LC points
xLC, yLC, wLC = LC.LC2Dpts(m,kappa,LCrange)
    
g = gcd(m[0],m[1])
e = np.mod(kappa[0]+kappa[1],2)

t = np.linspace(0, 2*np.pi, 1000)
x = np.zeros((g,np.shape(t)[0]))
y = np.zeros((g,np.shape(t)[0]))

for i in range(g):
    x[i,:] = np.cos(m[1]*t/g)
    y[i,:] = np.cos(m[0]*t/g+(2*i+e)*np.pi/m[1])
    x[i,:], y[i,:] = LC.norm_range(x[i,:],y[i,:],np.array([-1, 1, -1, 1]),LCrange)
 
#Prepare plot for spectral index sets
R = LC.LC2Dmask(m,kappa,0)
NoLC = np.count_nonzero(R)
gamma = np.zeros((NoLC,2))
ii = 0

for i in range(m[0]+1):
    for j in range(m[1]+1):
        if (R[i,j] > 0):
            gamma[ii,:] = [i,j]
            ii = ii+1
            
#Plot
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10, 5))

for i in range(g):
    ax1.plot(x[i,:],y[i,:],color=(183/255,207/255,246/255),linewidth=2,zorder=1)
  
ax1.scatter(x=xLC, y=yLC, marker='o', edgecolor='k', color = (65/255,105/255,225/255), s = 80, zorder = 2)
ax1.set_title('Lissajous-Chebyshev nodes')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_xlim(xmin=LCrange[0]-0.1,xmax=LCrange[1]+0.1)
ax1.set_ylim(ymin=LCrange[2]-0.1,ymax=LCrange[3]+0.1)

ax2.scatter(x=gamma[:,0], y=gamma[:,1], marker='o', edgecolor='k', color = (181/255,22/255,33/255), s = 80)
ax2.set_title('Spectral index sets')
ax2.set_xlabel('$\gamma_1$')
ax2.set_ylabel('$\gamma_2$')
fig.tight_layout()  
plt.show()