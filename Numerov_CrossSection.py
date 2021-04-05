import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.integrate import simps

from BesselFunctions import n_l, j_l

N=1000

sig = 3.18
eps = 5.9

coeff = 0.36 # hbar^2/2m in eps*sig^2

def inter(x):
    return 4*((1/x)**(12.)-(1/x)**(6.))
    

def WFsolveScattering(E, l):
    # Initialising arrays for the potential and for k-squared term
    Psi=np.zeros(N)
    v=np.zeros(N)
    ks=np.zeros(N)

    xmax=18.
    h=xmax/N
    x=np.linspace(0.5,xmax,N)

    for i in range(1,N): # define potential and k(x) squared factor
        v[i]=inter(x[i])
        ks[i]= (1./coeff)*(E-v[i]-l*(l + 1.)*coeff/(x[i]*x[i]))
            
    #### Initialise array for the function and prepare the initial condition
    Psi[0]= np.e**(- (2/5)*np.sqrt(1/coeff)/(x[0]**5))
    Psi[1]= np.e**(- (2/5)*np.sqrt(1/coeff)/(x[1]**5))

    ### Numerov Algorithm
    for i in range(2,N):
        Psi[i]=(2*Psi[i-1]*(1-5*(h*h/12)*(ks[i-1]))
                -Psi[i-2]*(1+(h*h/12)*(ks[i-2])))/(1+(h*h/12)*(ks[i]))
                
    kappa = Psi[N//2]/Psi[N//3]*x[N//3]/x[N//2]
    
    tanDelta = ((kappa* j_l(l,x[N//2]*np.sqrt(E/coeff))-j_l(l,x[N//3]*np.sqrt(E/coeff)))/
                (kappa* n_l(l,x[N//2]*np.sqrt(E/coeff))-n_l(l,x[N//3]*np.sqrt(E/coeff))))
    
    return x,Psi, np.arctan(tanDelta)
    
def CrossSect(E):
    XSum = 0.
    for l in range(0,7):
        Solve=WFsolveScattering(E, l)
        XSum = XSum + (2*l+1)*pow(np.sin(Solve[2]),2)
    return (sig*sig*4*np.pi/(E/(coeff)))*XSum
    
    
EnergyRange = np.linspace(0.01,0.59,100)
Xsect = np.zeros(100)

for i in range(0,100):
    Xsect[i]=CrossSect(EnergyRange[i])

plt.figure(figsize=(7,7))
plt.plot(EnergyRange*eps,Xsect/(sig*sig))
plt.show()
