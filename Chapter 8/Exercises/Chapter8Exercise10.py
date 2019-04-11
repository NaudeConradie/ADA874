#!/usr/bin/env python
# coding: utf-8

# In[26]:


import numpy as np
import os

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import MDS


# In[2]:


PROJECT_ROOT_DIR = r"C:\Users\19673418\Desktop\ADA874\Chapter 8\Exercises"
CHAPTER_ID = "Chapter8DR"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


# In[3]:


try:
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1)
    mnist.target = mnist.target.astype(np.int64)
except ImportError:
    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('MNIST original')


# In[4]:


x = mnist["data"]
y = mnist["target"]


# In[5]:


x_train = mnist['data'][:60000]
y_train = mnist['target'][:60000]

x_test = mnist['data'][60000:]
y_test = mnist['target'][60000:]


# In[6]:


tsne = TSNE(n_components = 2, random_state = 42)


# In[7]:


x_test_redux = tsne.fit_transform(x_test)


# In[18]:


plt.figure(figsize = (12, 10))
plt.scatter(x_test_redux[:, 0], x_test_redux[:, 1], c = y_test, cmap = "gist_rainbow")
plt.colorbar()
plt.axis('off')
plt.title("t-SNE MNIST Scatter Plot", fontsize = 14)
save_fig("t-SNE MNIST Scatter Plot")
plt.show()


# In[19]:


pca = PCA(n_components=2)


# In[20]:


x_test_redux_pca = pca.fit_transform(x_test)


# In[21]:


plt.figure(figsize = (12, 10))
plt.scatter(x_test_redux_pca[:, 0], x_test_redux_pca[:, 1], c = y_test, cmap = "gist_rainbow")
plt.colorbar()
plt.axis('off')
plt.title("PCA MNIST Scatter Plot", fontsize = 14)
save_fig("PCA MNIST Scatter Plot")
plt.show()


# In[23]:


lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)


# In[24]:


x_test_redux_lle = lle.fit_transform(x_test)


# In[25]:


plt.figure(figsize = (12, 10))
plt.scatter(x_test_redux_lle[:, 0], x_test_redux_lle[:, 1], c = y_test, cmap = "gist_rainbow")
plt.colorbar()
plt.axis('off')
plt.title("LLE MNIST Scatter Plot", fontsize = 14)
save_fig("LLE MNIST Scatter Plot")
plt.show()


# In[27]:


mds = MDS(n_components=2, random_state=42)


# In[28]:


x_test_redux_mds = mds.fit_transform(x_test[:1000])


# In[29]:


plt.figure(figsize = (12, 10))
plt.scatter(x_test_redux_mds[:, 0], x_test_redux_mds[:, 1], c = y_test[:1000], cmap = "gist_rainbow")
plt.colorbar()
plt.axis('off')
plt.title("MDS MNIST Scatter Plot", fontsize = 14)
save_fig("MDS MNIST Scatter Plot")
plt.show()


# In[ ]:




