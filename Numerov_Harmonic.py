from Numerov import *

## DEFAULT number of points
N=1000

## Define the Potential
harm="0.5*x*x" # Define Harmonic Potential

def Harmonic_Simulation(): # Plot the first 5 solutions (POINT 1)
    
    xgen=np.arange(-6,6,0.01)
    vgen=eval(harm,{"x": xgen})

    EEn=EnergyGuess(0.2,5,harm,N) # Find the eigenvalues

    plt.figure(figsize=(7,5))
    plt.xlabel('x',size =12)
    plt.title('Harmonic Oscillator', loc='left', weight='black', size=14
    , color = 'darkgoldenrod')
    plt.title('Energies and eigenfunctions', loc='right',
    color='grey', style='italic', size=12)

    for i in range(0,5): # Plot the wave functions
        Solve=WFsolve(EEn[i,0],harm,N)
        
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
        EnergyTable[3*i:3*(i+1),0:3]=EnergyGuess(0.3,10,harm,N,True,1.*i,3)
        
    ind=np.argsort(EnergyTable[:,0])
    EnergyTable=EnergyTable[ind]
    print("List of energy levels")
    print(tabulate(EnergyTable,floatfmt=(".6f",".0f",".0f"),headers=["Energy","n", "l"]))

def Check_N_Dependence():
    Points = [500,1000,2000,4000,8000,10000,20000,40000]
    Tab=np.zeros((8,2))
    for i in range(0,8):
        En=EnergyGuess(1.4,1.6,harm,Points[i])
        Tab[i]=[Points[i],abs(En[0,0]-1.5)]
    print(tabulate(Tab,floatfmt=(".0f",".8f"),headers=["Num","err"]))
    
#Check_N_Dependence()
#Radial_Simulation()
#Harmonic_Simulation()
