import numpy as np 
import pandas as pd
#%%
def fx(x):
    return x**2

def gx(x):
    return (x-2)**2

def bin2num (p):
        r = int(str(int("".join(map(str, p)))),2)
        D= XL+((XU-XL)/((2**(bits/2))-1)) * r
        return(D)
#%%
popln = 10
bits = 8
XU = 2
XL =-2
p= [] ; X= []  
for i in range(popln):
   a = np.random.choice([0, 1], size=bits)
   p.append(a)
        
D=[] #convert binary no. to decimal no with given range 
for i in range(popln):
    r=bin2num(p[i])
    D.append(r)  
    
#%%
fitness1 =[] ; fitness2 =[]
for i in np.arange(popln):
    F1= fx(D[i])
    F2= gx(D[i])
    fitness1.append(F1)
    fitness2.append(F2)

#D = pd.DataFrame(D)
fitness1 = pd.DataFrame(fitness1)
fitness2 = pd.DataFrame(fitness2)

#D.columns = ["energy"]
fitness1.columns = ["fitness1"]
fitness2.columns = ["fitness2"]

fitness = pd.concat([fitness1,fitness2],axis=1)
#%%
rank1=[]
for i in np.arange((len(p)-1)):
    if fitness.iloc[i,0]< fitness.iloc[i+1,0] and fitness.iloc[i,1]< fitness.iloc[i+1,1]:
        R = fitness.iloc[i,:]
        rank1.append(R)
rank1 = pd.DataFrame(rank1)
#    Rank1.append(rank1.iloc[0,:])
#    for i in np.arange(len(rank1)-1):
#        if rank1.iloc[0,0]< rank1.iloc[i+1,0] and rank1.iloc[0,1]< rank1.iloc[i+1,1]:
#            R = rank1.iloc[i+1,:]
#            Rank1.append(R)

#rank1=[];Rank1=[]
#if fitness.iloc[0,0]< fitness.iloc[1,0] and fitness.iloc[0,1]< fitness.iloc[1,1]:
#        R = fitness.iloc[0,:]
#        rank1.append(R)
#        #rank1 = pd.DataFrame(rank1)
#else:
#    rank1.append(fitness.iloc[1,:])
#
#rank1 = pd.DataFrame(rank1)  
#    
#Rank1.append(rank1)
#for i in np.arange(2,(len(p)-2)):
#    if rank1.iloc[0,0]< fitness.iloc[i,0] and rank1.iloc[0,1]< fitness.iloc[i,1]:
#        R = fitness.iloc[i,:]
#        Rank1.append(R)
#Rank1 = pd.DataFrame(Rank1)
#    