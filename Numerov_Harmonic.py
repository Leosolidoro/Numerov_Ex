from Numerov import *

## DEFAULT number of points
N=10000

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
def NumCheck_and_Radial():

    ENEfile = open("./Energy_Levels.txt","w")
##############
######## CHECK ENERGY DEPENDENCE ON POINTS IN THE MESH ######
###############
    Points = [500,1000,2000,4000,8000,10000,20000,40000]
#
    ENEfile.write("\n")
    ENEfile.write("############################\n")
    ENEfile.write("#### ENERGY EIGENVALUES ####\n")
    ENEfile.write("####   DEPENDENCE ON    ####\n")
    ENEfile.write("#####  NUMBER OF POINTS #####\n")
    ENEfile.write("############################\n")
    ENEfile.write("\n")
    ENEfile.write("Points \t | En- 1.5 | \n")
    ENEfile.write("------------------ \n")
    Tab=np.zeros((8,2))
    for i in range(0,8):
        En=EnergyGuess(1.4,1.6,harm,Points[i])
        ENEfile.write("%.0f \t  %f \n"%(Points[i],abs(En[0,0]-1.5)))

#################
##### FIND ENERGY LEVELS FOR 3 HARMONIC OSCILLATOR
#################

    EnTab=np.zeros((9,3))
    ENEfile.write("\n")
    ENEfile.write("############################\n")
    ENEfile.write("#### ENERGY EIGENVALUES ####\n")
    ENEfile.write("####   FOR THE RADIAL   ####\n")
    ENEfile.write("###  HARMONIC OSCILLATOR ###\n")
    ENEfile.write("############################\n")
    ENEfile.write("\n")
    ENEfile.write("En. \t \t n \t l \n")
    ENEfile.write("------------------ \n")
    
    for i in range(0,3):
        EnTab[3*i:3*(i+1),0:3]=EnergyGuess(0.3,10,harm,N,True,1.*i,3)
    ind=np.argsort(EnTab[:,0])
    EnTab=EnTab[ind]
    
    for i in range(0,9):
        ENEfile.write("%f \t  %.0f \t %.0f \n"%(EnTab[i,0],EnTab[i,1],EnTab[i,2]))
