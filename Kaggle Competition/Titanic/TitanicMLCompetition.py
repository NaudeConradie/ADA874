#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import sys
#!{sys.executable} -m pip install numpy

import numpy as np
import pandas as pd
import os

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.impute import SimpleImputer
from sklearn_pandas import CategoricalImputer

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

from sklearn.linear_model import LinearRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


# In[2]:


train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")


# In[3]:


train_data.head()


# In[4]:


train_data.info()


# In[5]:


medianAge = np.nanmedian(train_data["Age"])
medianAge


# In[6]:


num_attribs = ["Age", "SibSp", "Parch", "Fare"]
cat_attribs = ["Pclass", "Sex", "Embarked"]


# In[7]:


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values


# In[8]:


num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', SimpleImputer(strategy = "median")),
        ('std_scaler', StandardScaler()),
    ])


# In[9]:


cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('imputer', CategoricalImputer(strategy = "most_frequent")),
        ('encoder', OneHotEncoder(sparse = False)),
    ])


# In[10]:


pre_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])


# In[11]:


x_train = pre_pipeline.fit_transform(train_data)
x_test = pre_pipeline.fit_transform(test_data)


# In[12]:


y_train = train_data["Survived"]


# In[13]:


titanic_lr = LinearRegression()
titanic_lr.fit(x_train, y_train)


# In[14]:


lr_score = cross_val_score(titanic_lr, x_train, y_train, cv = 10)
lr_score.mean()


# In[15]:


titanic_svm = SVC(gamma = "auto")
titanic_svm.fit(x_train, y_train)


# In[16]:


svm_score = cross_val_score(titanic_svm, x_train, y_train, cv = 10)
svm_score.mean()


# In[17]:


titanic_rfc = RandomForestClassifier(n_estimators = 100, random_state = 42)
titanic_rfc.fit(x_train, y_train)


# In[18]:


rfc_score = cross_val_score(titanic_rfc, x_train, y_train, cv = 10)
rfc_score.mean()


# In[19]:


titanic_rfg = RandomForestRegressor(n_estimators = 100, random_state = 42)
titanic_rfg.fit(x_train, y_train)


# In[20]:


rfg_score = cross_val_score(titanic_rfg, x_train, y_train, cv = 10)
rfg_score.mean()


# In[21]:


titanic_dtr = DecisionTreeRegressor()
titanic_dtr.fit(x_train, y_train)


# In[22]:


dtr_score = cross_val_score(titanic_dtr, x_train, y_train, cv = 10)
dtr_score.mean()


# In[23]:


y_pred = titanic_svm.predict(x_test)
y_pred


# In[24]:


pd.DataFrame(y_pred).to_csv(r"C:\Users\19673418\Desktop")


# In[ ]:




