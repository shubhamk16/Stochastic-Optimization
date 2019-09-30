import numpy as np 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#%% 
def fitness (path):
    accuracy=[]
    for i in np.arange(len(path)):
        model = RandomForestClassifier(random_state=42 ,n_estimators=20,n_jobs=-1)
        model.fit( X_train[:,path[i]],y_train)
        y_pred = model.predict(X_test[:,path[i]])
        acc= accuracy_score(y_test,y_pred)
        accuracy.append(acc)  
    m = np.max(accuracy)   
    max = np.argmax(accuracy)
    BestAttributes = path[max]
    return(BestAttributes,m)
    
#***#****#****#****#****#***#****#***#****#****#*****
    
def phero_inc(Best_Attr, Ph):
    Ph = Ph* 0.8
    for i in np.arange(len(Best_Attr)):
        q= (Ph[Best_Attr[i]]*1.2) / 0.8
        Ph[Best_Attr[i]] =q 
    return(Ph)

#***#****#****#****#****#***#****#***#****#****#*****

def ACO (Ph):
    path =[]
    for i in np.arange(ant):
        tmpPh =np.copy(Ph)
        cityA=[]
        A = np.random.randint(0,3)
        cityA.append(A)
        for i in np.arange(subset-1):
            r = np.random.uniform(0,1)        
            if r < q0:
                tmpPh[A]=0
                A = np.argmax(tmpPh)
                cityA.append(A)
            else:
                tmpPh[A]=0
                prob = tmpPh / (sum(tmpPh))
                np.random.choice(range(4),p=prob)
                cityA.append(A)     
        path.append(cityA)
    return (path)
#%% 
X = load_iris().data
y = load_iris().target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

#%%
q0 = 0.7
subset = 3
ant = 4
max_iter = 100
phero = np.random.uniform(0.1,1,size=4)
#%%
for i in np.arange(max_iter):
    path  = ACO(phero)
    Best_Attr = fitness(path)[0]
    phero = phero_inc(Best_Attr,phero)
    Attribtes = ((np.argsort(phero).tolist())[::-1])[:subset]
    
print("The Best Attributes are:",Attribtes)
print("The Best accuracy is:",(fitness(path)[1])*100)


