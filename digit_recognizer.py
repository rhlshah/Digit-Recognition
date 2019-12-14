#Import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Read the Data
df = pd.read_csv('train.csv')
op=pd.read_csv('test.csv')

data = df.values
X_test=op.values

X_train=data[:,1:]
Y_train=data[:,0]

#Code the Algorithm

def dist(x1,x2):
    return np.sqrt(sum((x1-x2)**2))

def knn(X,Y,queryPoint,k=5):
    
    vals = []
    m = X.shape[0]
    
    for i in range(m):
        d = dist(queryPoint,X[i])
        vals.append((d,Y[i]))
        
    
    vals = sorted(vals)
    # Nearest/First K points
    vals = vals[:k]
    
    vals = np.array(vals)
    
    #print(vals)
    
    new_vals = np.unique(vals[:,1],return_counts=True)
    #print(new_vals)
    
    index = new_vals[1].argmax()
    pred = new_vals[0][index]
    
    return pred
    
    
def prediction(X_train,Y_train,X_test):
    m=X_test.shape[0]
    
    for i in range(m):
        pred=knn(X_train,Y_train,X_test[i])
        print(i,",",int(pred))



#Predict the output
prediction(X_train,Y_train,X_test)
