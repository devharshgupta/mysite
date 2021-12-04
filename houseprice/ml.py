#!/usr/bin/env python
# coding: utf-8

# In[54]:

# import os
from sklearn import metrics
import keras
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import sweetviz as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# get_ipython().run_line_magic('matplotlib', 'inline')

# cwd = os.getcwd()
# print(cwd)

data = pd.read_csv('train.csv')

data.head()


data.info()


# In[58]:


data.shape


# In[59]:


data.columns


# In[60]:


data.describe()


# In[61]:


data["UNDER_CONSTRUCTION"].value_counts()


# In[62]:


data["RERA"].value_counts()


# In[63]:


data["BHK_NO."].value_counts()


# In[64]:


data["READY_TO_MOVE"].value_counts()


# In[65]:


data["RESALE"].value_counts()


# In[66]:


data.isna().sum().sum()  # to check if any Null value exist


# In[67]:


data['BHK_OR_RK'].value_counts()


# In[68]:


sns.countplot("BHK_OR_RK", data=data)


# here, we cans see that bhk_or_rk class is highly unbalanced as BHK has around 29000 entries but Rk has only 24. so, we can drop this coloumn

# In[69]:


fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)

sns.countplot("BHK_OR_RK", data=data, ax=axes[0, 0])
sns.countplot("RESALE", data=data, ax=axes[0, 1])
sns.countplot("READY_TO_MOVE", data=data, ax=axes[0, 2])
sns.countplot("RERA", data=data, ax=axes[1, 0])
sns.countplot("UNDER_CONSTRUCTION", data=data, ax=axes[1, 1])


#

# In[70]:


#!pip install sweetviz


# In[71]:


report = st.analyze(data)
# report.show_html("./report.html")


# In[72]:


data["CITY"] = data["ADDRESS"].apply(lambda x: x.split(",")[1])
data["LOCALITY"] = data["ADDRESS"].apply(lambda x: x.split(",")[0])
data


# In[73]:


data.drop(['ADDRESS', 'BHK_OR_RK'], axis=1, inplace=True)


# In[74]:


data


# In[75]:


#from sklearn.preprocessing import LabelEncoder

#le=LabelEncoder()##


# In[76]:


# data['CITY']=le.fit_transform(data['CITY'])


# In[77]:


# data['LOCALITY']=le.fit_transform(data['LOCALITY'])


# In[78]:


data.head()


# In[79]:


data.drop(['POSTED_BY'], axis=1, inplace=True)


# In[80]:


data.head()


# In[81]:


plt.plot(data['SQUARE_FT'])


# In[82]:


plt.figure(figsize=(9, 9))
sns.heatmap(data.corr(), annot=True)


# under construction and ready to move are highly correlated

# In[83]:


data.drop(['READY_TO_MOVE', 'CITY', 'LOCALITY'], axis=1, inplace=True)


# In[84]:


data.head()


# In[85]:


X = data[['UNDER_CONSTRUCTION', 'RERA', 'BHK_NO.',
          'SQUARE_FT', 'RESALE', 'LONGITUDE', 'LATITUDE']]
Y = data[['TARGET(PRICE_IN_LACS)']]


# In[86]:


X.head(-1)


# In[87]:


Y.head(-1)


# In[88]:


X.info()


# In[89]:


X_train, X_test, y_train, y_test = train_test_split(
    X, Y, train_size=0.8, random_state=50)


# In[90]:


model = RandomForestRegressor()

model.fit(X_train, y_train)


y_pred = model.predict(X_test)


r2_score(y_test, y_pred)*100


# In[103]:


# In[112]:


model1 = keras.models.Sequential()

model1.add(keras.layers.Dense(7, activation='relu', input_shape=(7,)))
model1.add(keras.layers.Dense(7, activation='relu'))
model1.add(keras.layers.Dense(7, activation='relu'))
model1.add(keras.layers.Dense(7, activation='relu'))
model1.add(keras.layers.Dense(1))


model1.compile(optimizer='adam', loss='mean_squared_error')


model1.fit(X, Y, epochs=30)


model1.summary()


# In[106]:


print('Mean Absolute Error: {:.2f}'.format(
    metrics.mean_absolute_error(y_test, y_pred)))
print('Mean Squared Error: {:.2f}'.format(
    metrics.mean_squared_error(y_test, y_pred)))
print('Root Mean Squared Error: {:.2f}'.format(
    np.sqrt(metrics.mean_squared_error(y_test, y_pred))))
print('Variance score is: {:.2f}'.format(
    metrics.explained_variance_score(y_test, y_pred)))


# In[107]:


accuracy = model1.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy))

test_data = np.array([0, 1, 2, 929.921143, 1, 28.642300, 77.344500])
print(model.predict(test_data.reshape(1, 7)))


test_data = np.array([0, 1, 2, 929.921143, 1, 28.642300, 77.344500])
print(model1.predict(test_data.reshape(1, 7), batch_size=1))


test_data = np.array([0, 0, 2, 1022.641509, 1, 26.928785, 75.828002])
print(model.predict(test_data.reshape(1, 7)))


test_data = np.array([0, 0, 2, 1022.641509, 1, 26.928785, 75.828002])
print(model1.predict(test_data.reshape(1, 7), batch_size=1))


def Pridict_price(UNDER_CONSTRUCTION, RERA, BHK_NO, SQUARE_FT, RESALE, LONGITUDE, LATITUDE):

    test_data = np.array([UNDER_CONSTRUCTION, RERA, BHK_NO,
                         SQUARE_FT, RESALE, LONGITUDE, LATITUDE])
    return model1.predict(test_data.reshape(1, 7), batch_size=1)
