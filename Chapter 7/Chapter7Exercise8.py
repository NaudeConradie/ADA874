#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os

from sklearn.model_selection import train_test_split

from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.svm import LinearSVC
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score


# In[2]:


try:
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1)
    mnist.target = mnist.target.astype(np.int64)
except ImportError:
    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('MNIST original')


# In[3]:


x_trainval, x_test, y_trainval, y_test = train_test_split(mnist.data, mnist.target, test_size = 10000, random_state = 42)
x_train, x_val, y_train, y_val = train_test_split(x_trainval, y_trainval, test_size = 10000, random_state = 42)


# In[4]:


rf_clf = RandomForestClassifier(n_estimators = 10, random_state = 42)
et_clf = ExtraTreesClassifier(n_estimators = 10, random_state = 42)
svm_clf = SVC(max_iter = 100, random_state = 42, kernel = 'linear', probability = True)


# In[5]:


rf_clf.fit(x_train, y_train)


# In[6]:


et_clf.fit(x_train, y_train)


# In[7]:


#svm_clf.fit(x_train, y_train)


# In[8]:


hv_clf = VotingClassifier(
        estimators=[('rf', rf_clf), ('et', et_clf)],
        voting='hard'
)

hv_clf.fit(x_train, y_train)


# In[9]:


sv_clf = VotingClassifier(
        estimators=[('rf', rf_clf), ('et', et_clf)],
        voting='soft'
)

sv_clf.fit(x_train, y_train)


# In[10]:


for clf in (rf_clf, et_clf, hv_clf, sv_clf):
    y_pred = clf.predict(x_val)
    print(clf.__class__.__name__, accuracy_score(y_val, y_pred))


# In[11]:


for clf in (hv_clf, sv_clf):
    y_pred = clf.predict(x_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

