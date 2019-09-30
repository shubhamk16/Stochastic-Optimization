#Program for feature selection using Genetic Algorithm for Breast Cancer Dataset.
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import copy
import warnings
warnings.filterwarnings("ignore")

#%%
#Import Dataset
data = pd.read_csv("data.csv")

target=data["diagnosis"]
target.replace('M',0,inplace=True)
target.replace('B',1,inplace=True)
data = data.drop(data.columns[[0,1,32]], axis=1)
#data.isnull().sum()

#%% Population Generation
popln = 10 ;Pc = 0.7 ; Pm = 1/popln ;bits = len(data.columns) ; Max_itr = 50

pop=[]
for i in range(popln):
   a = np.random.choice([0, 1], size=bits)
   pop.append(a)

#%%

acc=[]; best_acc=[]; attri=[]

for k in range(Max_itr): 
    ACC =[] ; FIT =[]
    indx = [] ; D=[]
    for i in np.arange(len(pop)):   #============Fitness function===========#
        ind = list(np.where(pop[i]==1))
        indx.append(ind)
        I = indx[0][0]
        
        test = data[data.columns[list(I)]]
        X_train,X_test,y_train,y_test = train_test_split(test,target,test_size = 0.2,random_state = 42)
        model = RandomForestClassifier()    #============ RandomForest Model===========#
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        m_acc = accuracy_score(y_test,y_pred)

        ACC.append(m_acc) 
        c = np.linspace(0.2,1,5)
        fit=[]
        for j in np.arange(len(c)):     
            fitnes = m_acc - (j*len(I) / len(data.columns))
            fit.append(fitnes)
            F = np.max(fit)
        FIT.append(F)
        
    best_fit=[]         #============Tournament selection===========#
    random= np.random.randint(0, 10, size=(popln,2))
    for k in np.arange(popln):
        maximum = min(FIT[random[k][0]],FIT[random[k][1]])
        index = FIT.index(maximum)
        best_fit.append(index)

    Best_population=[]           #============ Best Chromosomes ===========#
    for m in np.arange(popln):
        B = pop[best_fit[m]]
        Best_population.append(B)

    random= np.random.randint(0, 10, size=(np.int(popln*0.5) ,2))
    cross_vec=[]            #============ Cross-Over ===========#
    for i in np.arange(len(random)):
        C1 = Best_population[random[i][0]]
        C2 = Best_population[random[i][1]]
        r = np.random.uniform(0,1,size=(1,bits))
        if np.any(r < Pc):
            z = r[r < Pc]
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
    
    for i in np.arange(popln):          #============ Mutation ===========#
        r = np.random.uniform(0,1,size=(1,bits))
        if np.any(r < Pm):
            z = r[r < Pm]
            r.tolist(); z.tolist()
            v= np.where(r==z[0])
            w= v[1][0]
            if cross_vec[i][w] == 1:
                cross_vec[i][w] = 0
            else:
                cross_vec[i][w] = 1
        else:   
            cross_vec[i]=cross_vec[i]
    cv=cross_vec 
    m_acc =[] ; m_FIT =[]
    m_indx = [] ; m_D=[]
    for i in np.arange(len(cv)): 
        m_ind = list(np.where(cv[i]==1))
        m_indx.append(m_ind)
        m_I = m_indx[0][0]
        
        m_test = data[data.columns[list(m_I)]]
        Xm_train,Xm_test,ym_train,ym_test = train_test_split(m_test,target,test_size = 0.2,random_state = 42)
        m_model = RandomForestClassifier()
        m_model.fit(Xm_train,ym_train)
        ym_pred = m_model.predict(Xm_test)
        mm_acc = accuracy_score(ym_test,ym_pred)
        m_acc.append(mm_acc) 
        m_c = np.linspace(0.2,1,5)
        m_fit=[]
        for j in np.arange(len(m_c)):     
            m_fitnes = mm_acc - (j*len(m_I) / len(data.columns))
            m_fit.append(m_fitnes)
            m_F = np.max(m_fit)
        m_FIT.append(m_F)
    best_ACC = np.max(m_FIT)    
    best_acc_ind = np.argmax(m_FIT)
    T_ind = m_indx[best_acc_ind]
    attri.append(T_ind)
    best_acc.append(best_ACC)
    acc.append(best_acc_ind)

best_attri_no = np.argmax(best_acc) 
                                    #============ Best Features ===========#
best_attri = attri[best_attri_no]
feat = data[data.columns[list(best_attri[0])]]
features = feat.columns

print("Best Features - %s"%list(features))
#%%