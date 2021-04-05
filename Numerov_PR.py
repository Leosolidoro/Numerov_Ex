import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.integrate import simps
from tabulate import tabulate

###############################################
###### NUMEROV METHOD FOR SCHROEDINGER EQ #####
###############################################

## DEFAULT number of points and error on energy's determination
N=10000
err=0.000001

## Define the harmonic Potential
def Potential(pot,x):
    return 0.5*x*x

######## NUMEROV SOLVING ALGORITHM ########
##### Solving the numerical euquation #####
###########################################
## E : Energy
## N : Number of Points in the mesh
## RAD : True for Central Potential problems
## l : Ang. Mom. quantum number
####
# Returns:  - Numerical Wave Function:
#                      x-axis, Wave Function (x, Psi)
#           - Value of the last point (Psi[N-1])
#           - Normalised WF for plotting pouposes (NPSI)
#           - Number of nodes of the function (nodes)
#
############################################
    
def WFsolve(E, N, RAD=False,l=0.):
    # Initialising arrays for the potential and for k-squared term
    v=np.zeros(N)
    ks=np.zeros(N)
    nodes = 0
    
     ## Define the boundaries of integration sufficiently
     ## far from the classical avoided bounds. Then initilise
     ## the mesh and fill v and ks

    if (RAD):
        xmax=np.sqrt(2*E)+3.0
        h=xmax/N
        x=np.linspace(0.,xmax,N)
    else:
        xmax=np.sqrt(2*E)+3
        h=2*xmax/N
        x=np.linspace((-1)*xmax,xmax,N)

    for i in range(1,N): # define potential and k(x) squared factor
        v[i]=Potential(x[i])
        if (RAD):
            ks[i]= 2.0*(E-v[i]-l*(l + 1.)/(2*x[i]*x[i]))
        else:
            ks[i]= 2.0*(E-v[i])
            
    #### Initialise array for the function and prepare the initial conditions
    Psi=np.zeros(N)
    if (RAD): # From the algebraic behaviour of the radial WF for r -> 0
        Psi[1]= h**(l+1) #pow(x[1],l)
    else: # close to zero for points far from classical allowed region
        Psi[1]=0.0000001
        ### Numerov Algorithm
    for i in range(2,N):
        Psi[i]=(2*Psi[i-1]*(1-5*(h*h/12)*(ks[i-1]))
                -Psi[i-2]*(1+(h*h/12)*(ks[i-2])))/(1+(h*h/12)*(ks[i]))
        if ((Psi[i]*Psi[i-1]<0) or (Psi[i]==0)): ## Counting the nodes as to
            nodes = nodes + 1                    ## evaluate the principal quantum number n

    area = simps(pow(Psi,2),dx=h) # Evaluate the area
    NPsi=pow(Psi/np.sqrt(area),2)+E # Normalised and shifted wave function TO PLOT
    
    return x,Psi,Psi[N-1],NPsi,nodes
    
######## BISECTION ALGORITHM #########
## Better approx. for Eigenenergies ##
######################################
## EnDW, EnUP: lower and upper limit for the energies
##              given by the corse scan in EnergyGuess
## N, RAD, L as in WFsolve
####
## Returns:  - Best Energy estimation (meanE)
##           - Number of nodes, i.e. principal q.n. (nodes)
##           - Associated error (errE)
##
######################################
    
def Bisect_Energy(EnDW,EnUP,N,RAD,l):
    Psi0=WFsolve(EnDW,N,RAD,l)
    Psi1=WFsolve(EnUP,N,RAD,l)
    
    condition=True
    while condition:
        Ei=(EnUP+EnDW)/2.
        Psii=WFsolve(Ei,N,RAD,l)
        if (Psii[2]==0):
            EnDW,EnUP=Ei,Ei
            break
        if (Psii[2]*Psi0[2]<0):
            EnUP=Ei
            Psi1=Psii
        else:
            EnDW=Ei
            Psi0=Psii
        if (abs(EnUP-EnDW)<err):
            break
    
        meanE= (EnUP+EnDW)/2
        errE = abs(EnUP-EnDW)/2
        nodes = int(np.minimum(Psi0[4],Psi1[4]))
    return meanE,nodes,l

######## EIGENVALUES ESTIMATION ########
##### Find energies which produces #####
#####   correct bound. condition    ####
########################################
## down, up : lower and upper limit for the scan
## limLevels : max. numbers of eigenvalues to find
## N, RAD, l: as in WFsolve
####
# Returns:  - Sorted array of Eigenvalues (EnergyArray) :
#                      ||Energy  0 | n0 | l0 ||
#                      ||Energy  1 | n1 | l1 || En.1> En.0
#
############################################
def EnergyGuess(down, up, N, RAD=False,l=0., limLevels=10,):
    # Initialise the arrays
    EnergyArray=np.zeros((limLevels,3)) # matrix of energy levels
    EnRange=np.arange(down,up,0.02) # List of energy to check
    EnPsi=np.zeros(2) # Array with last point two functions with energies in EnRange
    levels = 0 # counter for number of levelsof levels
    
    PP=WFsolve(EnRange[0]-0.1,N,RAD,l) # First guess to begin the cycle

    EnPsi[0]=PP[2]
    EnDW,EnUP=EnRange[0]-0.1,0. # first upper and lower bounds

    for En in EnRange:
        Psi=WFsolve(En,N,RAD,l)
        EnPsi[1]=Psi[2]
        EnUP=En
        if (EnPsi[0]*EnPsi[1]<0): # If sign changes, start fine scanning w/ Bisect
            guess=Bisect_Energy(EnDW,EnUP,N,RAD,l)
            EnergyArray[levels,0:3]=guess
            levels = levels +1
            
            if (levels == limLevels):
                break
            
        EnDW=En
        EnPsi[0]=EnPsi[1]
        
    np.sort(EnergyArray,axis=0)
    return EnergyArray
    

def Harmonic_Simulation(): # Plot the first 5 solutions (POINT 1)
    xgen=np.arange(-6,6,0.01)
    vgen=Potential(xgen)

    EEn=EnergyGuess(0.2,5,N) # Find the eigenvalues

    plt.figure(figsize=(7,5))
    plt.xlabel('x',size =12)
    plt.title('Harmonic Oscillator', loc='left', weight='black', size=14
    , color = 'darkgoldenrod')
    plt.title('Energies and eigenfunctions', loc='right',
    color='grey', style='italic', size=12)

    for i in range(0,5): # Plot the wave functions
        Solve=WFsolve(EEn[i,0],N)
        
        plt.hlines(EEn[i,0],-7,7,linestyle='dashed',colors='black',linewidth=1.)
        
        plt.fill_between(Solve[0],Solve[3],EEn[i,0],color='goldenrod')
        plt.plot(Solve[0],Solve[3],color='crimson')
        plt.text(3.8,EEn[i,0]+0.1,r'$E_{} = {:.3f}$'.format(i,EEn[i,0]))

    plt.plot(xgen,vgen,color='dodgerblue')
    plt.ylim(0,5.5)
    plt.xlim(-5.5,5.5)
    plt.savefig("./Harmonic.png",dpi=200)
    plt.show()
    
# Print list of EigenEnergies on the terminal (POINT 2)
def Radial_Simulation():
    EnergyTable=np.zeros((9,3))
    
    for i in range(0,3):
        EnergyTable[3*i:3*(i+1),0:3]=EnergyGuess(0.3,10,N,True,1.*i,3)
        
    ind=np.argsort(EnergyTable[:,0])
    EnergyTable=EnergyTable[ind]
    print("List of energy levels")
    print(tabulate(EnergyTable,floatfmt=(".6f",".0f",".0f"),headers=["Energy","n", "l"]))

def Check_N_Dependence():
    Points = [500,1000,2000,4000,8000,10000,20000,40000]
    Tab=np.zeros((8,2))
    for i in range(0,8):
        En=EnergyGuess(1.4,1.6,Points[i])
        Tab[i]=[Points[i],abs(En[0,0]-1.5)]
    print(tabulate(Tab,floatfmt=(".0f",".8f"),headers=["Num","err"]))
    
#Check_N_Dependence()
#Radial_Simulation()
Harmonic_Simulation()
