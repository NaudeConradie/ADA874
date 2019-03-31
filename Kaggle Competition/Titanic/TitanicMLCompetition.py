#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import sys
#!{sys.executable} -m pip install sklearn-pandas

import numpy as np
import pandas as pd
import os

from scipy.stats import randint

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
from sklearn.model_selection import RandomizedSearchCV


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


j_SibSp, j_Parch = [list(num_attribs).index(col) for col in ("SibSp", "Parch")]


# In[8]:


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values


# In[9]:


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_relatives = True):
        self.add_relatives = add_relatives
    def fit(self, X, y = None):
        return self
    def transform(self, X, y=None):
        if self.add_relatives:
            Relatives = X[:, j_SibSp] + X[:, j_Parch]
            return np.c_[X, Relatives]


# In[10]:


fancy_title_list = ["Sir.", "Lady.", "Count.", "Countess.", "Duke.", "Duchess.", "M.", "Mlle.", "\""]
female_title_list = ["Miss.", "Mrs."]
child_title_list = ["Master."]


# In[11]:


def find_string(X, string_list):
    found_strings = [None]*len(X)
    for string_el in range(0, len(X)):
        if any(x in X[string_el] for x in string_list):
            found_strings[string_el] = 1
        else:
            found_strings[string_el] = 0
    return found_strings


# In[12]:


class StringFinder(BaseEstimator, TransformerMixin):
    def __init__(self, find_strings = True):
        self.find_strings = find_strings
    def fit(self, X, y = None):
        return self
    def transform(self, X, y = None):
        if self.find_strings:
            FancyTitle = find_string(X, fancy_title_list)
            FemaleTitle = find_string(X, female_title_list)
            ChildTitle = find_string(X, child_title_list)
            return np.c_[FancyTitle, FemaleTitle, ChildTitle]


# In[13]:


num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', SimpleImputer(strategy = "median")),
        ('attr_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])


# In[14]:


cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('imputer', CategoricalImputer(strategy = "most_frequent")),
        ('encoder', OneHotEncoder(sparse = False)),
    ])


# In[15]:


str_pipeline = Pipeline([
        ('selector', DataFrameSelector("Name")),
        ('str_finder', StringFinder()),
    ])


# In[16]:


pre_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
        ("str_pipeline", str_pipeline),
    ])


# In[17]:


x_train = pre_pipeline.fit_transform(train_data)
x_test = pre_pipeline.fit_transform(test_data)


# In[18]:


y_train = train_data["Survived"]


# In[19]:


titanic_dtr = DecisionTreeRegressor(random_state = 42)
titanic_dtr.fit(x_train, y_train)


# In[20]:


dtr_score = cross_val_score(titanic_dtr, x_train, y_train, cv = 10)
dtr_score.mean()


# In[21]:


titanic_rfg = RandomForestRegressor(n_estimators = 100, random_state = 42)
titanic_rfg.fit(x_train, y_train)


# In[22]:


rfg_score = cross_val_score(titanic_rfg, x_train, y_train, cv = 10)
rfg_score.mean()


# In[23]:


titanic_lr = LinearRegression()
titanic_lr.fit(x_train, y_train)


# In[24]:


lr_score = cross_val_score(titanic_lr, x_train, y_train, cv = 10)
lr_score.mean()


# In[25]:


titanic_rfc = RandomForestClassifier(n_estimators = 100, random_state = 42)
titanic_rfc.fit(x_train, y_train)


# In[26]:


rfc_score = cross_val_score(titanic_rfc, x_train, y_train, cv = 10)
rfc_score.mean()


# In[27]:


y_pred_rfc = titanic_rfc.predict(x_test)
len(y_pred_rfc)


# In[28]:


submission = test_data["PassengerId"]
submission = pd.concat([submission, pd.DataFrame(y_pred_rfc)], axis=1)
submission.columns = ["PassengerId",  "Survived"]


# In[29]:


submission.to_csv("submissionRFC.csv", index = False)


# In[30]:


param_grid = [
        {'n_estimators': [75, 100, 200], 'max_features': [4, 5, 6]},
        {'bootstrap': [False], 'n_estimators': [100, 200], 'max_features': [4, 5, 6]},
    ]

titanic_rfc = RandomForestClassifier(random_state = 42)
grid_search = GridSearchCV(titanic_rfc, param_grid, cv = 5, scoring = 'neg_mean_squared_error')


# In[31]:


grid_search.fit(x_train, y_train)


# In[32]:


grid_search.best_params_


# In[33]:


grid_search.best_estimator_


# In[34]:


param_distrib = {'n_estimators': randint(low=1, high=1000), 'max_features': randint(low=1, high=8)}

titanic_rfc = RandomForestClassifier(random_state = 42)
random_search = RandomizedSearchCV(titanic_rfc, param_distributions = param_distrib, n_iter = 30, cv = 5, scoring = 'neg_mean_squared_error', random_state = 42)


# In[35]:


random_search.fit(x_train, y_train)


# In[36]:


random_search.best_params_


# In[37]:


random_search.best_estimator_


# In[38]:


titanic_rfc = RandomForestClassifier(max_features = 7, n_estimators = 21, random_state = 42)
titanic_rfc.fit(x_train, y_train)


# In[39]:


rfc_score = cross_val_score(titanic_rfc, x_train, y_train, cv = 10)
rfc_score.mean()


# In[40]:


y_pred_rfc = titanic_rfc.predict(x_test)
len(y_pred_rfc)


# In[41]:


submission = test_data["PassengerId"]
submission = pd.concat([submission, pd.DataFrame(y_pred_rfc)], axis=1)
submission.columns = ["PassengerId",  "Survived"]


# In[42]:


submission.to_csv("submissionRFC_impr.csv", index = False)


# In[43]:


titanic_svm = SVC(gamma = "auto")
titanic_svm.fit(x_train, y_train)


# In[44]:


svm_score = cross_val_score(titanic_svm, x_train, y_train, cv = 10)
svm_score.mean()


# In[45]:


y_pred_svm = titanic_svm.predict(x_test)
len(y_pred_svm)


# In[46]:


submission = test_data["PassengerId"]
submission = pd.concat([submission, pd.DataFrame(y_pred_svm)], axis=1)
submission.columns = ["PassengerId",  "Survived"]


# In[47]:


submission.to_csv("submissionSVM.csv", index = False)


# In[48]:


param_grid = {'C': [5, 10, 15], 'gamma': [0.01, 0.02, 0.03]}

titanic_svm = SVC()
grid_search = GridSearchCV(titanic_svm, param_grid, cv = 5, scoring = 'neg_mean_squared_error')


# In[49]:


grid_search.fit(x_train, y_train)


# In[50]:


grid_search.best_params_


# In[51]:


grid_search.best_estimator_


# In[52]:


titanic_svm = SVC(C = 10, gamma = 0.02)
titanic_svm.fit(x_train, y_train)


# In[53]:


svm_score = cross_val_score(titanic_svm, x_train, y_train, cv = 10)
svm_score.mean()


# In[54]:


y_pred_svm = titanic_svm.predict(x_test)
len(y_pred_svm)


# In[55]:


submission = test_data["PassengerId"]
submission = pd.concat([submission, pd.DataFrame(y_pred_svm)], axis=1)
submission.columns = ["PassengerId",  "Survived"]


# In[56]:


submission.to_csv("submissionSVM_impr.csv", index = False)


# In[ ]:




