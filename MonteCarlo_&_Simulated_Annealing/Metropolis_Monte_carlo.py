"""
  OPTIMIZATION CODE- Metropolis Monte_carlo.
"""
def rosenberg(x,y):
    return (1-x)**2 + 100*((y-x**2))**2

import numpy as np
x= np.floor(np.random.uniform(4,15))
y= np.floor(np.random.uniform(4,15))
E1 = rosenberg(x,y)

Energy=[]; X=[]; Y=[]
for i in np.arange(1500):
    h1,h2= np.random.uniform(-0.5,0.5),np.random.uniform(-0.5,0.5)
    newX= x+h1
    newY= y+h2
    X.append(newX)
    Y.append(newY)
    E2 = rosenberg(newX,newY)
    delta =E2-E1
    if delta <= 0 or np.random.uniform() < np.exp(-delta):
        x=newX
        y=newY
        Energy.append(E2) #Acccepted energy
    else:
        x=x
        y=y
        #Energy.append(E1)
Min = np.argmin(Energy)
x=X[Min];y = Y[Min]
print("The optimized value of x & y is:",x,y)
    
    

    
    
