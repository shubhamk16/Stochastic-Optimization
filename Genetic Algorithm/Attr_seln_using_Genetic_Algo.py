"""
Created on Mon Sep  9 21:02:29 2019

@author: Shubham Kothawade
"""
#%%
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
import copy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

#%% Load the dataset
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
#%% make the copy of original dataset,which will be need later
train_original=train.copy() 
test_original=test.copy()
#%%to check the attributes in train and test
train.columns
test.columns
#%%to check the datatypes and shape 
print(train.dtypes)
print(train.shape)

#%%
train.Dependents[train.Dependents=="3+"]= 4
test.Dependents[test.Dependents=="3+"]= 4
#%% preprocessing train data
#train['Loan_Status'].value_counts()
train.isnull().sum()
train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True) 
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)

train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)

#%% preprocessing test data
test.isnull().sum()
test['Gender'].fillna(train['Gender'].mode()[0], inplace=True) 
test['Married'].fillna(train['Married'].mode()[0], inplace=True) 
test['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)

test['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
test['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
test['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True) 
test['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)
test.isnull().sum()

#%% convert 
lm = LabelEncoder()
a  = ['Gender','Self_Employed','Property_Area','Married','Education']
for i in np.arange(len(a)):
    train[a[i]] = lm.fit_transform(train[a[i]])
    
for i in np.arange(len(a)):
    test[a[i]] = lm.fit_transform(test[a[i]])
    
#%%
train['Dependents'] = train['Dependents'].astype(np.int64)
test['Dependents'] = test['Dependents'].astype(np.int64)
#%%
train = train.drop('Loan_ID',1)
test = test.drop('Loan_ID',1)
#%%sklearn needs seperate target variables

X = train.drop('Loan_Status',1) 
y = train.Loan_Status.map({"Y":1,"N":0})
#%%
#from imblearn.over_sampling import SMOTE
#sm = SMOTE()
#X, y = sm.fit_sample(trainX, trainy)
#X = pd.DataFrame(X, columns = trainX.columns)

#%% feature engginering
#X['Total_Income']= (2*X['ApplicantIncome']* X['CoapplicantIncome']) /   (X['ApplicantIncome']+X['CoapplicantIncome'])
X['Total_Income']=(X['ApplicantIncome']*0.8)+(X['CoapplicantIncome']*0.2)
test['Total_Income'] = (test['ApplicantIncome']*0.8)+(test['CoapplicantIncome']*0.2)
#%%
#for i in np.arange(len(X)):
#    if X['Self_Employed'][i]=="No":
#        I=8
#    else:
#        I=9.25
#
#
#for i in np.arange(len(test)):
#    if test['Self_Employed'][i]=="No":
#        I1=8
#    else:
#        I1=9.25
I = 9
X['EMI'] = (X['LoanAmount']*1000*(I/1200)*(1 + 0.00833)**X['Loan_Amount_Term'])  / (((1 + 0.00833)**X['Loan_Amount_Term'])-1)
test['EMI'] = (test['LoanAmount']*1000*(I/1200)*(1 + 0.00833)**test['Loan_Amount_Term'])  / (((1 + 0.00833)**test['Loan_Amount_Term'])-1)

#%%
X['FOIR'] = X['EMI']/((X['Total_Income']*0.5))
test['FOIR'] = test['EMI'] / ((test['Total_Income']*0.5))
#
X['net_surplus'] = ((X['Total_Income']/2)-X['EMI'] ) / X['Total_Income']
test['net_surplus'] = ((test['Total_Income']/2)-test['EMI'] ) / test['Total_Income']
#
X['Act_HomePrice'] = (X['LoanAmount']*1000)/ 0.8
test['Act_HomePrice'] = (test['LoanAmount']*1000)/ 0.8

X['Golden_ratio'] = X['EMI']/ X['Total_Income']
test['Golden_ratio'] = test['EMI']/ test['Total_Income']
#
#%%

#X= X.drop(['Gender','Married','Education','ApplicantIncome','CoapplicantIncome','Self_Employed','Dependents','Loan_Amount_Term','EMI','Total_Income','LoanAmount'], axis=1) 
#test= test.drop(['Gender','Married','Education','ApplicantIncome','CoapplicantIncome','Self_Employed','Dependents','Loan_Amount_Term','EMI','Total_Income','LoanAmount'], axis=1) 
#train=train.drop(['Gender','ApplicantIncome','CoapplicantIncome'], axis=1) 
#test=test.drop(['Gender','ApplicantIncome','CoapplicantIncome'], axis=1) 



#%%
#X=train.drop('Loan_Status',1) 

#%%sklearn needs seperate target variables
#X = train.drop('Loan_Status',1) 
#y = train.Loan_Status.map({"Y":1,"N":0})

#%% split the traindata into train and test
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2)#, random_state=42)

#%%
#model = RandomForestClassifier(n_jobs=-1)
#model.fit(X_train,y_train)
#acc = model.score(X_test,y_test)
#pd.Series(model.feature_importances_,index=X_train.columns).sort_values(ascending=False)
#%%#X= X.drop(['Gender','Married','Education','ApplicantIncome','CoapplicantIncome','Self_Employed','Dependents','Loan_Amount_Term','EMI','Total_Income','LoanAmount'], axis=1) 

#X_train=X_train.drop(['Gender','Education','Self_Employed','ApplicantIncome','CoapplicantIncome'],axis=1)
#X_test=X_test.drop(['Gender','Education','Self_Employed','ApplicantIncome','CoapplicantIncome'],axis=1)

#%% feature selection using genetic algo.
popln = 30
bits = X_train.shape[1]
XL = 1
XU=bits
pc = 0.75
pm  = 0.2#1/popln #popln = 20 ;pc = 0.7 ; pm = 1/popln ;bits =4 
#%%
def cv(f):
    model = RandomForestClassifier(random_state=4,min_samples_split=5 ,n_estimators=350,n_jobs=-1,min_samples_leaf=2,criterion='gini')
    model.fit(X_train.iloc[:,f],y_train)
    #y_pred = model.predict(X_test.iloc[:,f])
    acc = model.score(X_test.iloc[:,f],y_test)
    return (acc)

def nonzero_indices(u):
        for i in np.arange(popln):
            nonzero= np.nonzero(u[i])
            index_attr= nonzero[0].tolist()
            indices.append(index_attr)
        return(indices)  
#%%Generate population with random no. betn 0 & 1 
p= [] ; indices =[]
for i in range(popln):
   a = np.random.choice([0,1], size=bits)
   p.append(a)     
        #%% finding function values
#for U in range(10): 
    
indices = nonzero_indices(p)

fitness =[]
for i in np.arange(popln):
    F= cv(indices[i])
    #fit = F -  (len(indices[i]) / X_train.shape[1])
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
indices2=[]
indices2 = nonzero_indices(p)

fitness2 =[]
for i in np.arange(popln):
    F= cv(indices2[i])
    #fit = F -  (len(indices[i]) / X_train.shape[1])
    fitness2.append(F)
#minFitness =np.argmin(Mut_fitness)
bestAttributes = [indices[np.where(fitness==max(fitness))[0][0]]]
best_Acc = max(fitness2)
print("\nAns ==>\nThe optimized value of x is:",best_Acc)

#%%
New_data = X.iloc[:,bestAttributes[0]]
new_test = test.iloc[:,bestAttributes[0]]

model = RandomForestClassifier(random_state=4,max_depth= 10,min_samples_split=5 ,n_estimators=350,n_jobs=-1,class_weight = 'balanced')
model.fit(New_data,y)
y_pred = model.predict(new_test)
#prob1 = model.predict_proba(test)
#%%
#from sklearn.linear_model import LogisticRegression 
#model = LogisticRegression() 
#model.fit(New_data,y)
#y_pred = model.predict(new_test)
#%%
pred = ["Y" if y_pred[i]==1 else "N" for i in range(len(y_pred))]


#%%
pd.DataFrame({'Loan_ID':test_original.Loan_ID,'Loan_Status': pred}).set_index('Loan_ID').to_csv('14sep2.csv')
pd.Series(model.feature_importances_,index=New_data.columns).sort_values(ascending=False)
