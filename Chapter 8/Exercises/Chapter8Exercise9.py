#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import time

from sklearn.ensemble import RandomForestClassifier

from sklearn.decomposition import PCA

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


x = mnist["data"]
y = mnist["target"]


# In[4]:


x_train = mnist['data'][:60000]
y_train = mnist['target'][:60000]

x_test = mnist['data'][60000:]
y_test = mnist['target'][60000:]


# In[5]:


rfc = RandomForestClassifier(n_estimators=10, random_state=42)


# In[6]:


t_start_rfc = time.time()
rfc.fit(x_train, y_train)
t_stop_rfc = time.time()


# In[7]:


t_train_rfc = t_stop_rfc - t_start_rfc
t_train_rfc


# In[8]:


y_pred = rfc.predict(x_test)
accuracy_score(y_test, y_pred)


# In[9]:


pca = PCA(n_components=0.95)


# In[10]:


t_start_pca_train = time.time()
x_train_redux = pca.fit_transform(x_train)
t_stop_pca_train = time.time()


# In[11]:


t_pca_train = t_stop_pca_train - t_start_pca_train
t_pca_train


# In[12]:


t_start_pca_test = time.time()
x_test_redux = pca.transform(x_test)
t_stop_pca_test = time.time()


# In[13]:


t_pca_test = t_stop_pca_test - t_start_pca_test
t_pca_test


# In[14]:


t_start_pcarfc = time.time()
rfc.fit(x_train_redux, y_train)
t_stop_pcarfc = time.time()


# In[15]:


t_train_pcarfc = t_stop_pcarfc - t_start_pcarfc
t_train_pcarfc


# In[16]:


y_pred_pca = rfc.predict(x_test_redux)
accuracy_score(y_test, y_pred_pca)


# In[17]:


t_pcarfc = t_pca_train + t_pca_test + t_train_pcarfc
t_pcarfc


# In[ ]:


# For the last run of the code:

# Time taken to train the Random Forest Classifier on an unreduced dataset:
# 3.951 s
# Accuracy Score of the Random Forest Classifier on an unreduced dataset:
# 0.9492

# Time taken to train the Random Forest Classifier on the reduced dataset:
# 9.741 s
# Total time taken to reduce the dataset and train the classifier:
# 16.978 s
# Accuracy Score of the Random Forest Classifier on an unreduced dataset:
# 0.9009

# Training on the reduced dataset takes more than twice as long, and if the
# time required to reduce the dataset is also taken into account, it takes
# more than four times as long.

