#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import sys
#!{sys.executable} -m pip install sklearn-pandas

import numpy as np
import pandas as pd
import os
import re

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt

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


ticket_text = re.findall(r"(?i)\b[a-z]+\b", train_data["Ticket"].to_string())
ticket_textset = set(ticket_text)
ticket_textset


# In[7]:


num_attribs = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
cat_attribs = ["Sex", "Embarked"]


# In[8]:


j_SibSp, j_Parch = [list(num_attribs).index(col) for col in ("SibSp", "Parch")]


# In[9]:


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values


# In[10]:


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_relatives = True):
        self.add_relatives = add_relatives
    def fit(self, X, y = None):
        return self
    def transform(self, X, y=None):
        if self.add_relatives:
            Relatives = X[:, j_SibSp] + X[:, j_Parch]
            return np.c_[X, Relatives]


# In[11]:


fancy_title_list = ["Sir.", "Lady.", "Count.", "Countess.", "Duke.", "Duchess.", "M.", "Mlle.", "\""]
female_title_list = ["Miss.", "Mrs.", "Lady.", "Countess.", "Duchess.", "Mlle."]
child_title_list = ["Master."]


# In[12]:


def find_string(X, string_list):
    found_strings = [None]*len(X)
    for string_el in range(0, len(X)):
        if any(x in X[string_el] for x in string_list):
            found_strings[string_el] = 1
        else:
            found_strings[string_el] = 0
    return found_strings


# In[13]:


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


# In[14]:


num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', SimpleImputer(strategy = "median")),
        ('attr_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])


# In[15]:


cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('imputer', CategoricalImputer(strategy = "most_frequent")),
        ('encoder', OneHotEncoder(sparse = False)),
    ])


# In[16]:


str_pipeline = Pipeline([
        ('selector', DataFrameSelector("Name")),
        ('str_finder', StringFinder()),
    ])


# In[17]:


pre_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
        ("str_pipeline", str_pipeline),
    ])


# In[18]:


x_train = pre_pipeline.fit_transform(train_data)
x_test = pre_pipeline.fit_transform(test_data)


# In[19]:


y_train = train_data["Survived"]


# In[20]:


titanic_dtr = DecisionTreeRegressor(random_state = 42)
titanic_dtr.fit(x_train, y_train)


# In[21]:


dtr_score = cross_val_score(titanic_dtr, x_train, y_train, cv = 10)
dtr_score.mean()


# In[22]:


titanic_rfg = RandomForestRegressor(n_estimators = 100, random_state = 42)
titanic_rfg.fit(x_train, y_train)


# In[23]:


rfg_score = cross_val_score(titanic_rfg, x_train, y_train, cv = 10)
rfg_score.mean()


# In[24]:


titanic_lr = LinearRegression()
titanic_lr.fit(x_train, y_train)


# In[25]:


lr_score = cross_val_score(titanic_lr, x_train, y_train, cv = 10)
lr_score.mean()


# In[26]:


titanic_rfc = RandomForestClassifier(n_estimators = 100, random_state = 42)
titanic_rfc.fit(x_train, y_train)


# In[27]:


rfc_score = cross_val_score(titanic_rfc, x_train, y_train, cv = 10)
rfc_score.mean()


# In[28]:


og_rfc_score = rfc_score.mean()


# In[29]:


y_pred_rfc = titanic_rfc.predict(x_test)
len(y_pred_rfc)


# In[30]:


submission = test_data["PassengerId"]
submission = pd.concat([submission, pd.DataFrame(y_pred_rfc)], axis=1)
submission.columns = ["PassengerId",  "Survived"]


# In[31]:


submission.to_csv("submissionRFC.csv", index = False)


# In[32]:


param_grid = [
        {'n_estimators': [75, 100, 200], 'max_features': [4, 5, 6]},
        {'bootstrap': [False], 'n_estimators': [100, 200], 'max_features': [4, 5, 6]},
    ]

titanic_rfc = RandomForestClassifier(random_state = 42)
grid_search = GridSearchCV(titanic_rfc, param_grid, cv = 5, scoring = 'neg_mean_squared_error')


# In[33]:


grid_search.fit(x_train, y_train)


# In[34]:


grid_search.best_params_


# In[35]:


grid_search.best_estimator_


# In[36]:


param_distrib = {'n_estimators': randint(low=1, high=1000), 'max_features': randint(low=1, high=8)}

titanic_rfc = RandomForestClassifier(random_state = 42)
random_search = RandomizedSearchCV(titanic_rfc, param_distributions = param_distrib, n_iter = 30, cv = 5, scoring = 'neg_mean_squared_error', random_state = 42)


# In[37]:


random_search.fit(x_train, y_train)


# In[38]:


random_search.best_params_


# In[39]:


random_search.best_estimator_


# In[40]:


titanic_rfc = RandomForestClassifier(max_features = 7, n_estimators = 436, random_state = 42)
titanic_rfc.fit(x_train, y_train)


# In[41]:


rfc_score = cross_val_score(titanic_rfc, x_train, y_train, cv = 10)
rfc_score.mean()


# In[42]:


new_rfc_score = rfc_score.mean()


# In[43]:


y_pred_rfc = titanic_rfc.predict(x_test)
len(y_pred_rfc)


# In[44]:


submission = test_data["PassengerId"]
submission = pd.concat([submission, pd.DataFrame(y_pred_rfc)], axis=1)
submission.columns = ["PassengerId",  "Survived"]


# In[45]:


submission.to_csv("submissionRFC_impr.csv", index = False)


# In[46]:


titanic_svm = SVC(gamma = "auto")
titanic_svm.fit(x_train, y_train)


# In[47]:


svm_score = cross_val_score(titanic_svm, x_train, y_train, cv = 10)
svm_score.mean()


# In[48]:


og_svm_score = svm_score.mean()


# In[49]:


y_pred_svm = titanic_svm.predict(x_test)
len(y_pred_svm)


# In[50]:


submission = test_data["PassengerId"]
submission = pd.concat([submission, pd.DataFrame(y_pred_svm)], axis=1)
submission.columns = ["PassengerId",  "Survived"]


# In[51]:


submission.to_csv("submissionSVM.csv", index = False)


# In[52]:


param_grid = {'C': [10, 15, 20], 'gamma': [0.01, 0.02, 0.03]}

titanic_svm = SVC()
grid_search = GridSearchCV(titanic_svm, param_grid, cv = 5, scoring = 'neg_mean_squared_error')


# In[53]:


grid_search.fit(x_train, y_train)


# In[54]:


grid_search.best_params_


# In[55]:


grid_search.best_estimator_


# In[56]:


titanic_svm = SVC(C = 15, gamma = 0.02)
titanic_svm.fit(x_train, y_train)


# In[57]:


svm_score = cross_val_score(titanic_svm, x_train, y_train, cv = 10)
svm_score.mean()


# In[58]:


new_svm_score = svm_score.mean()


# In[59]:


y_pred_svm = titanic_svm.predict(x_test)
len(y_pred_svm)


# In[60]:


submission = test_data["PassengerId"]
submission = pd.concat([submission, pd.DataFrame(y_pred_svm)], axis=1)
submission.columns = ["PassengerId",  "Survived"]


# In[61]:


submission.to_csv("submissionSVM_impr.csv", index = False)


# In[62]:


PROJECT_ROOT_DIR = r"C:\Users\19673418\Desktop\Repository\ADA874\Kaggle Competition\Titanic"
CHAPTER_ID = "Titanic"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


# In[63]:


models = ["Decision Tree Regressor", "Random Forest Regressor", "Linear Regression", "Random Forest Classifier", "Improved Random Forest Classifier", "Support Vector Machine", "Improved Support Vector Machine"]
model_pos = np.arange(len(models))
model_acc = [dtr_score.mean(), rfg_score.mean(), lr_score.mean(), og_rfc_score, new_rfc_score, og_svm_score, new_svm_score]
model_per = [x*100 for x in model_acc]
model_per


# In[64]:


plt.figure(figsize=(10, 5))
plt.barh(model_pos, model_per, align = 'center', alpha = 0.5)
plt.yticks(model_pos, models)
plt.xlabel('Accuracy (%)')
plt.title('Model Accuracies')
axes = plt.gca()
axes.set_xlim([0, 100])
save_fig("Model Accuracies")
plt.show


# In[ ]:




