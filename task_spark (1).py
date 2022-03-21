#!/usr/bin/env python
# coding: utf-8

# In[23]:


#Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[24]:


url = "http://bit.ly/w-data"
data = pd.read_csv(url)


# In[25]:


print(data.shape)
data.head()


# In[26]:


data.describe()


# In[27]:


data.info()


# In[28]:


data.plot(kind = 'scatter',x='Hours',y='Scores');
plt.show()


# In[29]:


data.corr(method = 'pearson')


# In[30]:


data.corr(method = 'spearman')


# In[31]:


hours = data['Hours']
scores = data['Scores']


# In[46]:


sns.distplot(hours)


# In[47]:


sns.distplot(scores)


# In[34]:


X = data.iloc[:, :-1].values
y = data.iloc[:, 1].values


# In[35]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=50)


# In[36]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)


# In[45]:


m=reg.coef_
c=reg.intercept_
line=m*X+c
plt.scatter(X,y)
plt.plot(X,line);
plt.show()


# In[39]:


y_pred = reg.predict(X_test)


# In[40]:


actual_predicted=pd.DataFrame({'Target':y_test,'Predicted':y_pred})
actual_predicted


# In[44]:


sns.set_style('whitegrid')
sns.distplot(np.array(y_test-y_pred))
plt.show()


# In[19]:


h=9.25
s=reg.predict([[h]])
print("If a student studies for {} hours per day he/she will score {} % in exam.".format(h,s))


# In[20]:


from sklearn import metrics
from sklearn.metrics import r2_score
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,y_pred))
print('R2 Score:',r2_score(y_test,y_pred))

