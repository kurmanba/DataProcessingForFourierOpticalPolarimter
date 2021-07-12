""" 
Main Python module for bivariate polynomial interpolation 
on general Lissajous-Chebyshev node points LC 
(C) P. Dencker and W. Erb 01.10.2016

This python module was written by W. Erb (01.10.2018)
"""

import numpy as np

def norm_range(x,y,raold,ranew):      
          x = (x - raold[0]) / (raold[1] - raold[0])
          y = (y - raold[2]) / (raold[3] - raold[2])
          x = (x*(ranew[1]-ranew[0])) + ranew[0]
          y = (y*(ranew[3]-ranew[2])) + ranew[2]
          return x, y
      
def T(N,x):
          y=np.cos(np.outer(np.arange(0, N+1, 1),np.arccos(x)))
          return y

def LC2Dpts(m,kappa,LCrange):      
          zx =  np.cos(np.linspace(0,1,m[0]+1)*np.pi)
          zy =  np.cos(np.linspace(0,1,m[1]+1)*np.pi)
          LC2, LC1 = np.meshgrid(zy,zx)
          W = np.ones((m[0]+1,m[1]+1))/m[0]/m[1]*2
          W[0,:] = W[0,:]/2
          W[m[0],:] = W[m[0],:]/2
          W[:,0] = W[:,0]/2
          W[:,m[1]] = W[:,m[1]]/2
          M2, M1 = np.meshgrid(np.arange(0,m[1]+1,1),np.arange(0,m[0]+1,1)) 
          findM = np.where(np.mod(M1+M2+kappa[0]+kappa[1]+1,2)==1)
          xLC = LC1[findM]
          yLC = LC2[findM]
          wLC = W[findM] 
          xLC, yLC = norm_range(xLC,yLC,[-1,1,-1,1],LCrange)
          return xLC, yLC, wLC
      
def LC2Dmask(m,kappa,average):
          M2, M1 = np.meshgrid(np.arange(0,m[1]+1,1),np.arange(0,m[0]+1,1)) 
          R = 1.0*(M1*m[1]+M2*m[0]<m[0]*m[1])
          if (average == 1):
              Req = 0.5*(M1*m[1]+M2*m[0]==m[0]*m[1])
              Req[0,m[1]] = 0.25
              Req[m[0],0] = 0.25  
              if (np.mod(kappa[0]+kappa[1],2) == 1) and (np.mod(m[0],2) == 0) and (np.mod(m[1],2) == 0):
                  Req[int(m[0]/2),int(m[1]/2)] = 0
          elif (average == 0):
              Req = np.multiply(1.0*(M1*m[1]+M2*m[0]==m[0]*m[1]),1.0*(M1/m[0] < M2/m[1]))
              Req[0,m[1]] = 0.5 
              if (np.mod(kappa[0]+kappa[1],2) == 0) and (np.mod(m[0],2) == 0) and (np.mod(m[1],2) == 0):
                  Req[int(m[0]/2),int(m[1]/2)] = 0.5         
          R = R + Req
          return R
      
def LC2DdatM(m,kappa,f,wLC):      
          M2, M1 = np.meshgrid(np.arange(0,m[1]+1,1),np.arange(0,m[0]+1,1)) 
          findM = np.where(np.mod(M1+M2+kappa[0]+kappa[1]+1,2)==1)
          G = np.zeros((m[0]+1,m[1]+1))
          G[findM] = f*wLC
          G = np.reshape(G,M1.shape)
          return G
      
def LC2Dcfsfft(m,kappa,G,average):
          Gh = np.fft.fft(G,2*m[0],0).real  
          Gh = Gh[0:m[0]+1,:]           
          Ghh = np.fft.fft(Gh,2*m[1],1).real      
          Ghh = Ghh[:,0:m[1]+1]
          M2, M1 = np.meshgrid(np.arange(0,m[1]+1,1),np.arange(0,m[0]+1,1)) 
          Alpha = (2-1.0*(M1<1))*(2-1.0*(M2<1))
          R = LC2Dmask(m,kappa,average)
          C = Ghh*Alpha*R
          return C

def LC2Dquad(C, m, LCrange):
          wx = np.zeros([m[0]+1,1])
          wy = np.zeros([m[1]+1,1])
          area = (LCrange[1]-LCrange[0])*(LCrange[3]-LCrange[2])
          wx[::2] = np.vstack(2/(1 - np.arange(0,m[0]+1,2)**2))
          wy[::2] = np.vstack(2/(1 - np.arange(0,m[1]+1,2)**2))
          Qf = np.sum(np.matmul(wx.T,C)*wy.T,1)*area/4
          return Qf      
      
def LC2Deval(C, m, x, y, LCrange):
          x,y = norm_range(x,y,LCrange,[-1, 1, -1, 1])
          Tx = T(m[0], x)
          Ty = T(m[1], y)
          Sf = np.sum(np.matmul(Tx.T,C)*Ty.T,1)
          return Sf
      
def testfun2D(x,y,n):
          if n == 1:
                  f1 = np.exp(np.sin(10*x)) + np.sin(6*np.exp(y))
                  f2 = np.sin(7*np.sin(x)) + np.sin(np.sin(8*y)) 
                  f3 = np.sin(x + y)+ 1/4*(x**2 + y**2)
                  f = f1 + f2 + f3
          elif n== 2:
                  f1 = np.exp(-(5-10*x)**2/2)+0.7*np.exp(-(5-10*y)**2/2)
                  f2 = 0.3*np.exp(-(5-10*x)**2/2)*np.exp(-(5-10*y)**2/2)
                  f = f1+f2
          elif n== 3:
                  f1 = np.exp(-0.05*np.sqrt((80*x-40)**2+(90*y-45)**2))
                  f2 = np.cos(0.3*np.sqrt((80*x-40)**2+(90*y-45)**2))
                  f = f1*f2
          elif n== 4:    
                  K = [1,0]
                  f = np.cos(K[0]*np.arccos(x))*np.cos(K[1]*np.arccos(y))
          elif n== 5:
                  f1 = .75 * np.exp(-(9 * x - 2) ** 2 / 4.0 - (9 * y - 2) ** 2 / 4.0)
                  f2 = .75 * np.exp(-(9 * x + 1) ** 2 / 49.0 - (9 * y + 1) / 10.0)
                  f3 = .5 * np.exp(-(9 * x - 7) ** 2 / 4.0 - (9 * y - 3) ** 2 / 4.0)
                  f4 = -.2 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)
                  f = f1+f2+f3+f4
          elif n== 6:
                  K1 = [1,3] 
                  K2 = [1,3]
                  f1 = np.cos(K1[0]*np.arccos(x))*np.cos(K1[1]*np.arccos(y))
                  f2 = np.cos(K2[0]*np.arccos(x))*np.cos(K2[1]*np.arccos(y))
                  f = f1*f2
          return f