# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np

glass= pd.read_csv("C:\\Users\\suraj baraik\\Desktop\\Data Science\\Suraj\\New folder (12)\\Module 18 Machine Learning K nearest Neighbour\\Assignment\\glass.csv")

glass["Type"].value_counts()

##Checking for the data distribution of the data
data = glass.describe()

## As, there is difference in the scale of the values, we normalise the data.

def norm_fumc(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)

norm = norm_fumc(glass.iloc[:,0:9])
glass1 = glass.iloc[:,9]
type(glass1)

##Splitting the data into train and test data using stratified sampling

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(norm,glass1,test_size = 0.4,stratify = glass1)

##Checking the distribution of the labels which are taken
glass["Type"].value_counts()
y_train.value_counts()
y_test.value_counts()

##Building the model
from sklearn.neighbors import KNeighborsClassifier as KN

model = KN(n_neighbors = 5)
model.fit(x_train,y_train)

##Finding the accuracy of the model on training data
train_accuracy = np.mean(model.predict(x_train)==y_train) ##70.8125%
 
##Accuracy on test data
test_accuracy = np.mean(model.predict(x_test)==y_test) ##60.465%

##Changing the K value

model2 = KN(n_neighbors = 9)
model2.fit(x_train,y_train)

##Accuracy on training data
train_two = np.mean(model2.predict(x_train)==y_train) ##65.625

##Accuracy on test data
test_two = np.mean(model2.predict(x_test)==y_test) ## 56.97

# creating empty list variable 
acc = []

# running KNN algorithm for 3 to 50 nearest neighbours(odd numbers) and 
# storing the accuracy values 
for i in range (4,30,1):
    model = KN(n_neighbors = i)
    model.fit(x_train,y_train)
    train_acc = np.mean(model.predict(x_train)==y_train)
    test_acc = np.mean(model.predict(x_test)==y_test)
    acc.append([train_acc, test_acc])

import matplotlib.pyplot as plt

##Training accuracy plot
plt.plot(np.arange(4,30,1),[i[0] for i in acc],'bo-')

##Test accuracy plot

plt.plot(np.arange(4,30,1),[i[1] for i in acc],'ro-')

plt.legend(["train","test"])

model3 = KN(n_neighbors = 6)
model3.fit(x_train,y_train)

pred_train = model3.predict(x_train)
cross_tab = pd.crosstab(y_train,pred_train)

train_accuracy = np.mean(pred_train == y_train)
##67.96%

pred_test = model3.predict(x_test)
cross_tab_test = pd.crosstab(y_test,pred_test)

test_accuracy=np.mean(pred_test ==y_test)
## 58.2%