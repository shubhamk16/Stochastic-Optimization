"""
  OPTIMIZATION CODE- Random walk.
"""
import numpy as np

def rosenberg(x,y):
    return (1-x)**2 + 100*((y-x**2))**2

x= np.floor(np.random.uniform(4,10))
y= np.floor(np.random.uniform(4,10))
fx = rosenberg(x,y)

Energy=[];X=[];Y=[]
for i in np.arange(1500):
    h= np.random.uniform(0,1)
    x=x-h
    y=y-h
    X.append(x)
    Y.append(y)
    f = rosenberg(x,y)
    Energy.append(f)
Min = np.argmin(Energy)
x=X[4];y = Y[4]
print("The optimized value of x & y is:",x,y)
    
    
