import numpy as np
import pandas as pd
max_iter = 50 ; q0  = 0.7
alpha = 1.1; beta = 1.2
cities=[1,2,3,4]
#%% function fro increasing pheromone
pheromone_inc=[]
def pher_inc(p_inc,Ph):
    Ph = 0.8 * Ph
    for i in np.arange(matrix.shape[0]):
        u= p_inc[:-1] ; v = p_inc[1:]
        q= ((Ph[u[i],v[i]])*1.4) / 0.8
        Ph[u[i],v[i]] = q
    return(Ph)
#%% function for calculating dist
def Dist(cityA):
    distance = []
    for i in np.arange(matrix.shape[0]):
        u= cityA[:-1] ; v = cityA[1:]
        d=dist[u[i],v[i]]
        distance.append(d)
        total_dist= sum(distance)
    return(total_dist)
#%%
dist = np.random.uniform(10,100,size=(4,4))
closeness = 1/dist
matrix = np.tril(closeness) + np.tril(closeness, -1).T
#or  matrix = (closeness + closeness.T) / 2
np.fill_diagonal(matrix,0)

pheromone = np.random.uniform(0.1,1,size=(4,4))
pheno_close_prod = np.multiply(pheromone,matrix)

sum_of_mat = sum(sum(pheno_close_prod ))
normalize_mat  = 1/sum_of_mat * (np.ones(matrix.shape))
probabilistic_mat = np.multiply(normalize_mat, pheno_close_prod)

probab_mat =np.copy(probabilistic_mat )
pheno_close = np.copy(pheno_close_prod)
#%%
shortest_dist = []
for i in np.arange(500):
    tot_dist=[]
    path = []
    for i in np.arange(4):
        A = np.random.randint(0,4)
        cityA=[]
        cityA.append(A)
           
        for i in np.arange(3):
            r = np.random.uniform(0,1)        
            if r < q0:
                c = np.argmax(pheno_close_prod[:,A])
                cityA.append(c)
                pheno_close_prod[:,A]=0
                pheno_close_prod[A]=0
                probabilistic_mat[:,A]=0
                probabilistic_mat[A]=0   
                A=c
            else:
                c = np.argmax(probabilistic_mat[:,A])
                cityA.append(c)
                pheno_close_prod[:,A]=0
                pheno_close_prod[A]=0
                probabilistic_mat[:,A]=0
                probabilistic_mat[A]=0   
                A=c 
        cityA.extend([cityA[0]])
        d = Dist(cityA)
        probabilistic_mat= np.copy(probab_mat)
        pheno_close_prod = np.copy(pheno_close)
        path.append(cityA)
        tot_dist.append(d)
    p_inc=path[np.where(tot_dist == min(tot_dist))[0][0]]
    pheromone=np.copy(pher_inc(p_inc,pheromone))
    shortest_dist.append(tot_dist)
       
      
    
