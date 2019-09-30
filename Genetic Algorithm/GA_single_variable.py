""" Optimization Code: Genetic algorithm """

import numpy as np
import copy
popln = int(input("please give no. of initial popln:"))
XL=int(input("please give lower range:")) 
XU=int(input("please give upper range:")) 
bits = int(input("please give no. of bits:"))
pc = float(input("probability of crossover:"))
pm  = 1/popln #popln = 20 ;pc = 0.7 ; pm = 1/popln ;bits =4 
#%%
def fun (x):
     return (x**2)

def bin2num (p):
        r = int(str(int("".join(map(str, p)))),2)
        D= XL+((XU-XL)/((2**bits)-1)) * r
        return(D)
#%%Generate population with random no. betn 0 & 1 
p= [] ; X= []  
for i in range(popln):
   a = np.random.choice([0, 1], size=bits)
   p.append(a)
for U in range(1000):    
    D=[] #convert binary no. to decimal no with given range 
    for i in range(popln):
        r=bin2num(p[i])
        D.append(r)       
    #%% finding function values
    fitness =[]
    for i in np.arange(popln):
        F= fun(D[i])
        fitness.append(F)
    #%%Tournament Selection
    Best_fit=[]
    random= np.random.randint(0, 10, size=(popln,2))
    for k in np.arange(popln):
        maximum = min(fitness[random[k][0]],fitness[random[k][1]])
        index = fitness.index(maximum)
        Best_fit.append(index)
    
    #%%finding Best chromosomes from original popln with help of tournament selection
    Best_population=[]
    for m in np.arange(popln):
        B = p[Best_fit[m]]
        Best_population.append(B)
    #%% crossover
    random= np.random.randint(0, 10, size=(np.int(popln*0.5) ,2))
    cross_vec=[]
    for i in np.arange(len(random)):
        C1 = Best_population[random[i][0]]
        C2 = Best_population[random[i][1]]
        r = np.random.uniform(0,1,size=(1,bits))
        if np.any(r < pc):
            z = r[r < pc]
            r.tolist(); z.tolist()
            v= np.where(r==z[0])
            w= v[1][0]
            Ori_C1 = C1.copy()
            C1[w+1:] = C2[w+1:]
            C2[w+1:] = Ori_C1[w+1:]
            cross_vec.append(C1)
            cross_vec.append(C2)
        else:
            C1=C1 ; C2=C2
            cross_vec.append(C1)
            cross_vec.append(C2)
    crossover = copy.deepcopy(cross_vec)
    
#%%   Mutation
    for i in np.arange(popln):
        r = np.random.uniform(0,1,size=(1,bits))
        if np.any(r < pm):
            z = r[r < pm]
            r.tolist(); z.tolist()
            v= np.where(r==z[0])
            w= v[1][0]
            if cross_vec[i][w] == 1:
                cross_vec[i][w] = 0
            else:
                cross_vec[i][w] = 1
        else:   
            cross_vec[i]=cross_vec[i]
    p=cross_vec # now mutated popln became new initial popln
#%% finding energy after 1000 iteration of mutated chromosome 
Mutated_Energy=[] 
for i in range(popln):
    r2=bin2num(p[i])
    Mutated_Energy.append(r2)
 
Mut_fitness =[]
for i in np.arange(popln):
    F1= fun(Mutated_Energy[i])
    Mut_fitness.append(F1)
minFitness =np.argmin(Mut_fitness)
best_Mut_X = Mutated_Energy[minFitness]
print("\nAns ==>\nThe optimized value of x is:",best_Mut_X)


