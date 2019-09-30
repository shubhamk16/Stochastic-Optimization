"""
  OPTIMIZATION CODE- Simulated annealing.
"""
def rosenberg(x,y):
    return (1-x)**2 + 100*((y-x**2))**2

import numpy as np
x= np.random.uniform(-5,5)
y= np.random.uniform(-5,5)
E1 = rosenberg(x,y)

Acp_energy=[];  Rej_energy=[]; X=[]; Y=[]
T = 1000
for i in np.arange(500):
    h1= np.random.uniform(-1,1)
    h2= np.random.uniform(-1,1)
    newX= x+h1
    newY= y+h2
    X.append(newX)
    Y.append(newY)
    E2 = rosenberg(newX,newY)
    delta =E2-E1
    if delta <= 0 or np.random.uniform(0,1) < np.exp(-delta/T):
        x=newX
        y=newY
        E1=E2
        Acp_energy.append(E2) #Acccepted energy
    else :
        x=x
        y=y
        Rej_energy.append(E1) #Rejected energy

m1=len(Acp_energy)
m2= len(Rej_energy) 
f = np.mean(Rej_energy)
To = -f /(np.log((0.95* (m1+m2) -m1)/m2))
print("Initaial temp:",To)
T=To

x= np.random.uniform(-5,5)
y= np.random.uniform(-5,5)
E1 = rosenberg(x,y)
energy =[]; X=[]; Y=[]

for i in np.arange(T):
    iter =1000
    for i in np.arange(iter):
        h1= np.random.uniform(-1,1)
        h2= np.random.uniform(-1,1)
        newX= x+h1
        newY= y+h2
        X.append(newX)
        Y.append(newY)
        E2 = rosenberg(newX,newY)
        delta =E2-E1
        if delta <= 0 or np.random.uniform() < np.exp(-delta/T):
            x=newX
            y=newY
            E1=E2
            energy.append(E2)
        else :
            energy.append(E2)
    
    if len(energy) >= 100 or iter >= 100:
        T = T*0.9
        if T<=1:
            break

Min = np.argmin(energy)
x=X[Min];y = Y[Min]

print("The value of rosenberg fun is:",rosenberg(x,y))

print("\nThe optimized value of x & y is:",x,y)

print("\nInitaial temp And Final Temp :",To ,T)
    
    

    
    
