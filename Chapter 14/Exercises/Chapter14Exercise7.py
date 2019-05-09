#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt

import random

import tensorflow as tf


# In[2]:


PROJECT_ROOT_DIR = r"C:\Users\19673418\Desktop\ADA874\Chapter 14\Exercises"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


# In[3]:


def reset_graph(seed = 42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


# In[4]:


Reber_grammar = [
    [("B", 1)],
    [("T", 2), ("P", 3)],
    [("S", 2), ("X", 4)],
    [("T", 3), ("V", 5)],
    [("X", 3), ("S", 6)],
    [("P", 4), ("V", 6)],
    [("E", None)],
]

emb_Reber_grammar = [
    [("B", 1)],
    [("T", 2), ("P", 3)],
    [(Reber_grammar, 4)],
    [(Reber_grammar, 5)],
    [("T", 6)],
    [("P", 6)],
    [("E", None)],
]


# In[5]:


Reber_chars = "BTPSXVE"


# In[6]:


def gen_Reber_string(grammar):
    
    state = 0
    output = []
    
    while state is not None:
        
        i = np.random.randint(len(grammar[state]))
        
        produced_grammar, state = grammar[state][i]
        
        if isinstance(produced_grammar, list):
            
            produced_grammar = gen_Reber_string(grammar = produced_grammar)
            
        output.append(produced_grammar)
    
    return "".join(output)


# In[7]:


for i in range(10):
    print(gen_Reber_string(emb_Reber_grammar), end = "\n")


# In[8]:


def gen_rand_string():
    
    I = np.random.randint(9, high = 20)
    
    return "".join(random.choice(Reber_chars) for i in range(I))


# In[9]:


for i in range(10):
    print(gen_rand_string(), end = "\n")


# In[10]:


def one_hot_enc_str_to_vec(string, n, chars = Reber_chars):
    
    char_to_i = {char: i for i, char in enumerate(chars)}
    
    output = np.zeros((n, len(chars)), dtype = np.int32)
    
    for i, char in enumerate(string):
        
        output[i, char_to_i[char]] = 1
        
    return output


# In[11]:


test_str = gen_Reber_string(emb_Reber_grammar)

one_hot_enc_str_to_vec(test_str, len(test_str))


# In[12]:


def gen_data(size):
    
    Reber_data = [gen_Reber_string(emb_Reber_grammar)
                  for i in range(size // 2)]
    
    rand_data = [gen_rand_string()
                 for i in range(size // 2)]
    
    data = Reber_data + rand_data
    
    n = max([len(string)
             for string in data])
    
    x = np.array([one_hot_enc_str_to_vec(string, n)
                  for string in data])
    
    l = np.array([len(string)
                  for string in data])
    
    y = np.array([[1] for i in range(len(Reber_data))] +
                 [[0] for i in range(len(rand_data))])
    
    rand_i = np.random.permutation(size)
    
    return x[rand_i], l[rand_i], y[rand_i]


# In[13]:


x_train, l_train, y_train = gen_data(10000)
x_test, l_test, y_test = gen_data(5000)


# In[14]:


reset_graph()


# In[15]:


n_i = len(Reber_chars)
n_n = 25
n_o = 1

learning_rate = 0.02
momentum = 0.95

x = tf.placeholder(tf.float32, [None, None, n_i], name = "x")
l = tf.placeholder(tf.int32, [None], name = "l")
y = tf.placeholder(tf.float32, [None, 1], name = "y")

gru_cell = tf.nn.rnn_cell.GRUCell(num_units = n_n)
o, states = tf.nn.dynamic_rnn(gru_cell, x, dtype = tf.float32, sequence_length = l)

logits = tf.layers.dense(states, n_o, name = "logits")

y_pred = tf.cast(tf.greater(logits, 0), tf.float32, name = "y_pred")
y_prob = tf.nn.sigmoid(logits, name = "y_prob")

x_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels = y, logits = logits)
loss = tf.reduce_mean(x_entropy, name = "loss")

optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate, momentum = momentum, use_nesterov = True)
training_op = optimizer.minimize(loss)

correct = tf.equal(y_pred, y, name = "correct")
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name = "accuracy")

init = tf.global_variables_initializer()
saver = tf.train.Saver()


# In[16]:


n_epochs = 25
batch_size = 50

with tf.Session() as tfs:
    
    init.run()
    
    for epoch in range(n_epochs):
        
        x_batches = np.array_split(x_train, len(x_train) // batch_size)
        l_batches = np.array_split(l_train, len(l_train) // batch_size)
        y_batches = np.array_split(y_train, len(y_train) // batch_size)
        
        for x_batch, l_batch, y_batch in zip(x_batches, l_batches, y_batches):
            loss_val, i = tfs.run(
                [loss, training_op],
                feed_dict = {x: x_batch, l: l_batch, y: y_batch}
            )
        
        acc_train = accuracy.eval(feed_dict = {x: x_batch, l: l_batch, y: y_batch})
        acc_val = accuracy.eval(feed_dict = {x: x_test, l: l_test, y: y_test})
        
        print("{:4d} Training Loss: {:.3f}, And Accuracy: {:.2f}%   Validation Accuracy: {:.2f}%".format(
            epoch, loss_val, 100*acc_train, 100*acc_val))
        saver.save(tfs, "./Reber_classifier")


# In[ ]:




