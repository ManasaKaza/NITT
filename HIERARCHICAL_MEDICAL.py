#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[3]:


data = pd.read_csv('medical.csv')
data


# In[4]:


data.isnull().sum()


# In[5]:


data = data.drop(['platelets','creatinine_phosphokinase'], axis = 1)


# In[6]:


plt.figure(1, figsize = (20 ,8))
sns.heatmap(data)
plt.show()


# In[7]:

import scipy.cluster.hierarchy as shc


# In[8]:


a = shc.linkage(data, method='average')  
plt.figure()
plt.title("Dendrograms")
# Dendrogram plotting using linkage matrix
dendrogram = shc.dendrogram(a)


# In[9]:


from sklearn.decomposition import PCA
reduced_data = PCA(n_components=2).fit_transform(data)
results = pd.DataFrame(reduced_data,columns=['pca1','pca2'])
a1 = shc.linkage(results, method='ward')
  
plt.figure()
dendrogram = shc.dendrogram(a1)


# In[10]:


from sklearn.cluster import AgglomerativeClustering
clus = AgglomerativeClustering(n_clusters = 5)
plt.figure(figsize =(6, 10))
plt.scatter(results['pca1'], results['pca2'],
           c = clus.fit_predict(results), cmap ='rainbow')
plt.show()


# In[11]:


from sklearn.metrics import silhouette_score

# Perform hierarchical clustering
Z = shc.linkage(data, method='average', metric='euclidean')  # Example parameters, adjust as needed

# Extract cluster labels
k = 3  # Example number of clusters, adjust as needed
labels = shc.fcluster(Z, k, criterion='maxclust')  # Adjust criterion as needed

# Calculate Silhouette Coefficient
silhouette_coefficient = silhouette_score(data, labels, metric='euclidean')  # Adjust metric as needed

print("Silhouette Coefficient:", silhouette_coefficient)


# In[12]:


from sklearn import metrics
fm=metrics.fowlkes_mallows_score(data['DEATH_EVENT'].astype(int),
                            labels.astype(int))
print("Fowlkes-Mallows Score:",fm)


# In[ ]:




