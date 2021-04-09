import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from BesselNeumann import n_l , j_l

N=10000

r_min = 0.5
hbar2m = 0.035 # h^2bar/2m * sig^2*eps

r = np.linspace(r_min,10.,N)

h = (10-r_min)/N

ks= np.zeros(N)

def inter_eff(r,l):
    return 4*(pow(r,-12.)-pow(r,-6.))+hbar2m*l*(l + 1)/(r*r)

    
def WaveFunction(E, l):
    Psi= np.zeros(N)
    
    for i in range(0,N):
        ks[i]=(E-inter_eff(r[i],l))/hbar2m
    
    Psi[0]= np.exp(-(2/5)/np.sqrt(hbar2m)*pow(r[0],-5))
    Psi[1]= np.exp(-(2/5)/np.sqrt(hbar2m)*pow(r[1],-5))
    
    for i in range(2,N):
        Psi[i]= (2*Psi[i-1]*(1 - 5*(h*h/12)*ks[i-1])
                 -Psi[i-2]*(1 + (h*h/12)*ks[i-2]))/(1 + (h*h/12)*ks[i])
    return Psi
    
def Cross_Comp(E,l_max):

    cross = 0.
    k = np.sqrt(E/hbar2m)
    lam = 2*np.pi/k
    
    r_1 = 5.0
    r_2 = r_1 + lam/4.
    
    N1 = int((r_1-r_min)/h)
    N2 = int((r_2-r_min)/h)
    
    for l in range(0,l_max+1):
        Psi = WaveFunction(E,l)
        kappa = (r_2/r_1)*(Psi[N1]/Psi[N2])
        Delta = np.arctan((kappa*j_l(l-1,k*r_2) - j_l(l-1,k*r_1))/
                    (kappa*n_l(l-1,k*r_2) - n_l(l-1,k*r_1)))
                    
        cross = cross + (2.*l + 1)*pow(np.sin(Delta),2)
    
    return (4*np.pi/(k*k))*cross , Delta
        
NUM = 100
Sigma = np.zeros(NUM)
Energy = np.linspace(0.01,0.6,NUM)

for i in range(0,NUM):
    Sigma[i] = Cross_Comp(Energy[i],6)[0]
    
plt.figure(figsize=(7,5))
plt.plot(Energy,Sigma,color='crimson')
plt.fill_between(Energy,Sigma,color='goldenrod')
plt.ylabel('$\sigma(E) \,\,[\sigma^2]$')
plt.xlabel('$E/\epsilon$')
plt.ylim(0,60)
plt.xlim(0.01,0.6)
plt.savefig("./Cross.png",dpi=200)
