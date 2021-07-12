""" 
Main test example for bivariate polynomial interpolation 
on general Lissajous-Chebyshev node points LC 

(C) P. Dencker and W. Erb 01.10.2016

This python implementation was written by W. Erb (01.10.2018)
"""

import LC2Ditp as LC
#from mpl_toolkits import mplot3d
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

#Set the parameters
LCrange = [0, 1, 0, 1]       #Range of Lissajous curves
m       = [28,29]            #Frequency parameters
kappa   = [0,0]              #Phase shift parameter

nofun   = 2                  #Number of test function
average = 0                  #Averaging on boundaries
Nd      = 100                #Discretization for plot

# Extract LC points
xLC, yLC, wLC = LC.LC2Dpts(m,kappa,LCrange)
NoLC = wLC.size

# Extract function values at LC points
fLC = LC.testfun2D(xLC,yLC,nofun)
# Generate data matrix at LC points
G = LC.LC2DdatM(m,kappa,fLC,wLC)
# Calculate the expansion coefficients
C = LC.LC2Dcfsfft(m,kappa,G,average)

#Evaluate the interpolation polynomial at new grid
[x,y] = np.meshgrid(np.linspace(LCrange[0],LCrange[1],Nd),np.linspace(LCrange[2],LCrange[3],Nd))   
Sflin = LC.LC2Deval(C,m,x.flatten(),y.flatten(),LCrange)   
SfLC  = LC.LC2Deval(C,m,xLC,yLC,LCrange)      
Sf    = np.reshape(Sflin,(Nd,Nd))  

#Evaluation of integral
Qf    = LC.LC2Dquad(C,m,LCrange);

# Calculate the errors
maxerror   = LA.norm(Sflin-LC.testfun2D(x.flatten(),y.flatten(),nofun),np.inf);
maxerrorLC = LA.norm(SfLC-fLC,np.inf);

# Display the results and plot the interpolant
if (maxerrorLC > 1e-12):
    print('Error: Interpolation not successful!')
else:
    print('Interpolation successful!')
    if (maxerror < 1e-12):
        print('The test function was reproduced exactly.\n')
    else:
        print('')

print('Number of interpolation points   : %23d' % NoLC)
print('Maximal error for approximation  : %23.18f' % maxerror)
print('Maximal error at LC points       : %23.18f \n' % maxerrorLC)
print('Integral of function over range  : %23.18f \n' % Qf)
 
fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection='3d')
ax.plot_surface(x, y, Sf, rstride=1, cstride=1,
                cmap='Wistia', edgecolor='none',zorder = 1)
ax.scatter(xLC, yLC, fLC, color = (0.8,0.8,0.8), edgecolor='k', linewidth=1, s = 50, zorder = 2)
ax.set_title('Polynomial interpolant on LC nodes',y=1.1)
ax.view_init(50, -120)