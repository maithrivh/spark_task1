#!/usr/bin/env python
# coding: utf-8




#Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Loading datasets
url = "http://bit.ly/w-data"
data = pd.read_csv(url)


print(data.shape)#returns a tuple, the elements of the tuple give the lengths of the array dimensions.
data.head()# prints the reshaped data

data.describe()
# describe() is used for calculating statistical data like percentile,mean,std of the numerical values of the Data Frame 

data.info()
#The info() method prints information about the DataFrame.


data.plot(kind = 'scatter',x='Hours',y='Scores');
plt.show()



data.corr(method = 'pearson')





data.corr(method = 'spearman')





hours = data['Hours']
scores = data['Scores']





sns.distplot(hours)




sns.distplot(scores)




X = data.iloc[:, :-1].values
y = data.iloc[:, 1].values





from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=50)





from sklearn.linear_model import LinearRegression
#linear_model is a class of the sklearn module if contain different functions for performing machine learning with linear models. 
reg = LinearRegression()
reg.fit(X_train, y_train)





m=reg.coef_
c=reg.intercept_
line=m*X+c
plt.scatter(X,y)
plt.plot(X,line);
plt.show()





y_pred = reg.predict(X_test)





actual_predicted=pd.DataFrame({'Target':y_test,'Predicted':y_pred})
actual_predicted





sns.set_style('whitegrid')
sns.distplot(np.array(y_test-y_pred))
plt.show()





h=9.25
s=reg.predict([[h]])
print("If a student studies for {} hours per day he/she will score {} % in exam.".format(h,s))




from sklearn import metrics
from sklearn.metrics import r2_score
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,y_pred))
print('R2 Score:',r2_score(y_test,y_pred))

