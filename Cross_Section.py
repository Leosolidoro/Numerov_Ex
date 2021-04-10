import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from BesselNeumann import n_l , j_l

N=1000
eps = 5.9
sig = 3.18

hbar2m = 0.035 # h^2bar/2m / sig^2*eps

def inter_eff(r,l): ## Lennard-Jones Potential
    return 4*(pow(r,-12.)-pow(r,-6.))+hbar2m*l*(l + 1)/(r*r)


def WaveFunction(E, l, r_min=0.5, r_max = 5): ## Numerov Method
    Psi= np.zeros(N)

    r = np.linspace(r_min,10.,N)

    h = (10-r_min)/N

    ks= np.zeros(N)

    for i in range(0,N):
        ks[i]=(E-inter_eff(r[i],l))/hbar2m

    Psi[0]= np.exp(-(2/5)/np.sqrt(hbar2m)*pow(r[0],-5))
    Psi[1]= np.exp(-(2/5)/np.sqrt(hbar2m)*pow(r[1],-5))

    for i in range(2,N):
        Psi[i]= (2*Psi[i-1]*(1 - 5*(h*h/12)*ks[i-1])
                 -Psi[i-2]*(1 + (h*h/12)*ks[i-2]))/(1 + (h*h/12)*ks[i])

    k = np.sqrt(E/hbar2m) # Wave vector Module
    lam = 2*np.pi/k # Sinusoid Wavelengh

    r_1 = r_max

    r_2 = r_1 + lam/4.
    if (lam > 20):
        r_2 = 8.0

    N1 = int((r_1-r_min)/h)
    N2 = int((r_2-r_min)/h)

    kappa = (r_2/r_1)*(Psi[N1]/Psi[N2])
    Delta = np.arctan((kappa*j_l(l,k*r_2) - j_l(l,k*r_1))/
                (kappa*n_l(l,k*r_2) - n_l(l,k*r_1)))

    return Psi, Delta

def Cross_Comp(E,l_max): ## Evaluate Cross Section

    kk = E/hbar2m
    cross = 0.

    for l in range(0,l_max+1):
        Psi = WaveFunction(E,l)
        Delta = Psi[1]

        cross = cross + (2.*l + 1)*pow(np.sin(Delta),2)

    return (4*np.pi/(kk))*cross


def Cross_Plot():
    NUM = 200
    Sigma = np.zeros(NUM)
    Energy = np.linspace(0.001,0.6,NUM)

    for i in range(0,NUM):
        Sigma[i] = Cross_Comp(Energy[i],6)


### Plotting stuff

    textstr = '\n'.join(
    (r'$H-Kr \,\, scattering$',r'Total Cross Section'))
    props=dict(boxstyle='round', facecolor='goldenrod', alpha=0.5)

    fig, ax1 = plt.subplots(figsize=(7,5))

    #ax1.set_facecolor('gold')
    ax1.plot(Energy,Sigma,color='crimson', linewidth=2.5)
    plt.fill_between(Energy,Sigma,color='goldenrod',alpha=0.8)

    ax1.text(0.95, 0.90, textstr, transform=ax1.transAxes, fontsize=14,weight='bold',
        verticalalignment='top', bbox=props,horizontalalignment = 'right')

    ax1.set_ylabel('$\sigma_{\mathrm{tot}}(E) / \sigma^2$', fontsize = 12)
    ax1.set_xlabel('$E/\epsilon$', fontsize = 12)
    plt.ylim(0,70)
    plt.xlim(0.001,0.6)
    def velocity(E):
        return 4.36*np.sqrt(E*eps)
    def Kinet(v):
        return 0.01*(v*v)/(19*eps)
    ax2 = ax1.secondary_xaxis('top', functions=(velocity,Kinet))
    ax2.set_xlabel(r'$g(E)\,\,[10^2\cdot m/s]$',fontsize = 12, weight='bold')
    plt.savefig("./Cross.png",dpi=200)

Cross_Plot()
