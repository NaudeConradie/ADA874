#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports

import numpy as np
import os

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt


# In[2]:


# Function to sort the MNIST dataset, as fetch_openml returns it unsorted

def sort_by_target(mnist):
    
    # The training and test sets are separately sorted
    reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1]
    reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[60000:])]))[:, 1]
    
    # The instances and their labels are sorted so that they still match
    mnist.data[:60000] = mnist.data[reorder_train]
    mnist.target[:60000] = mnist.target[reorder_train]
    mnist.data[60000:] = mnist.data[reorder_test + 60000]
    mnist.target[60000:] = mnist.target[reorder_test + 60000]


# In[3]:


# Fetch the MNIST dataset and sort it

try:
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    mnist.target = mnist.target.astype(np.int8) # fetch_openml returns the labels as strings, so we convert them to integers
    sort_by_target(mnist) # The dataset is sorted
except ImportError:
    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('MNIST original') # The old function fetch_mldata is used if fetch_openml fails

# The datasets characteristics are shown for inspection
mnist["data"], mnist["target"]


# In[4]:


x, y = mnist["data"], mnist["target"]


# In[ ]:


# The dataset is split into a training set and test set

x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]


# In[ ]:


# The training set is randomly shuffled

shuffle_index = np.random.permutation(60000)
x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier()
knn_clf.fit(x_train, y_train)


# In[ ]:


y_train_predict = knn_clf.predict(x_test)


# In[ ]:


from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_train_predict)

