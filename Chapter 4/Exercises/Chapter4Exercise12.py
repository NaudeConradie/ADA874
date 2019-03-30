#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Implement Batch Gradient Descent with early stopping for Softmax Regression (without using Scikit-Learn).

import numpy as np


# In[2]:


np.random.seed(42)
x = 2*np.random.rand(100, 1)
y = 4 + 3*x+np.random.randn(100, 1)


# In[3]:


x_b = np.c_[np.ones((100, 1)), x]


# In[4]:


eta = 0.1
n_i = 10
m = 100

np.random.seed(42)
theta = np.random.randn(2, 1)


# In[ ]:


for i in range(n_i):
    gradients = 2/m * x_b.T.dot(x_b.dot(theta) - y)
    theta = theta - eta*gradients


# In[5]:


minimum_val_error = float("inf")
best_epoch = None
best_model = None


# In[7]:


for epoch in range(1000):
    
    p_hat = exp(theta.T.dot(x_b))
    
    gradients = 1/m * sum()
    
    y_hat = np.argmax(theta.T.dot(x_b))
    
    if val_error < minimum_val_error:
        minimum_val_error = val_error
        best_epoch = epoch
        best_model = clone(sgd_reg)

