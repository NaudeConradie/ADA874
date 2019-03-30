#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Implement Batch Gradient Descent with early stopping for Softmax Regression (without using Scikit-Learn).

import numpy as np

from sklearn import datasets


# In[2]:


iris = datasets.load_iris()

x = iris["data"][:, (2, 3)]
y = iris["target"]


# In[3]:


m = 120


# In[4]:


x_train, x_test, y_train, y_test = x[:120], x[120:], y[:120], y[120:]


# In[5]:


np.random.seed(42)

shuffle_index = np.random.permutation(120)
x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]


# In[6]:


np.random.seed(42)

theta = np.random.randn(120, 2)


# In[7]:


s_k = theta.T.dot(x_train)
s_k


# In[8]:


p_hat = np.exp(s_k)/sum(np.exp(s_k))
p_hat


# In[9]:


y_hat = np.argmax(p_hat)
y_hat


# In[12]:


y_k = 1 if y_hat == 1 else 0

J = -1/m * sum(sum(y_k*np.log(p_hat)))
J


# In[ ]:


gradients = 

