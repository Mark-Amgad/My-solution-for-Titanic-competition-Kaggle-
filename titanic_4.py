# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 12:28:56 2020

@author: NEW TECH
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# get training and test datasets
dataset = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")










# get the critical features only
X = dataset.iloc[:,[4,5,2,6,7,9]].values
X_test = data_test.iloc[:,[3,4,1,5,6,8]].values

# My Independent varibale (Survived or not)
y = dataset.iloc[:,1].values

# Dealing with missing data by getting the mean value of Nan values
from sklearn.preprocessing import Imputer 
imp = Imputer(missing_values="NaN" , strategy="mean", axis=0)
imp = imp.fit(X[ : , 1:])
X[: ,1:] = imp.transform(X[:,1:])
X_test[ : ,1:] = imp.transform(X_test[ : ,1:])




# Encoding categorical data (Sex column)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
X_test[:,0] = labelencoder_X.fit_transform(X_test[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
X_test = onehotencoder.fit_transform(X_test).toarray()

# avoid dummy variables
X = X[:,1:]
X_test = X_test[:,1:]





# get my train and test sets
# this section was used when testing only
#from sklearn.cross_validation import train_test_split
#X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2, random_state = 0)


# feature scaling
from sklearn.preprocessing import StandardScaler
scaling = StandardScaler()
scaling.fit(X)
X = scaling.transform(X)
X_test = scaling.transform(X_test)



# SVM model
from sklearn.svm import SVC
k_svc_model = SVC(kernel="rbf",random_state=0)
k_svc_model.fit(X,y)







# my predictions
y_pred = k_svc_model.predict(X_test)

# reshaping in order to merge
y_pred = y_pred.reshape((len(y_pred),1))
ids = data_test.iloc[:,0].values.reshape((len(data_test),1))

# preparing to submit
res = np.concatenate((ids,y_pred),axis=1)
res = pd.DataFrame(data=res,columns=["PassengerId","Survived"])

#final move ..
res.to_csv(r"G:\Kaggle\Titanic\final_4.csv",index=False)


# **** this section was used in testing ****
# Confusion matrix to check if my model fit well or not
#from sklearn.metrics import confusion_matrix
#CM = confusion_matrix(y_test,y_pred)
#print(CM)