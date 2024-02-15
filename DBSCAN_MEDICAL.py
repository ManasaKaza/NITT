#!/usr/bin/env python
# coding: utf-8

# In[47]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# In[48]:


data = pd.read_csv('medical.csv')
data


# In[49]:


data.isnull().sum()


# In[64]:


data.nunique()


# In[50]:


from scipy.stats import mode

def eucledian(p1,p2):
    dist = np.sqrt(np.sum((p1-p2)**2))
    return dist
 
def predict(x_train, y , x_input, k):
    op_labels = []

    for item in x_input: 
        point_dist = []
        for j in range(len(x_train)): 
            distances = eucledian(np.array(x_train[j,:]) , item) 

            point_dist.append(distances) 
        point_dist = np.array(point_dist) 

        dist = np.argsort(point_dist)[:k] 

        labels = y[dist]
        lab = mode(labels) 
        lab = lab.mode[0]
        op_labels.append(lab)
 
    return op_labels


# In[66]:


from sklearn.neighbors import NearestNeighbors
ns = 3
nbrs = NearestNeighbors(n_neighbors=ns).fit(data)
distances, indices = nbrs.kneighbors(data)
distanceDec = sorted(distances[:,ns-1], reverse=False)
plt.plot(distanceDec)


# In[67]:


from kneed import KneeLocator
kneedle = KneeLocator(x = range(1, len(indices)+1), y = distanceDec, S = 1.0, 
                      curve = "concave", direction = "increasing", online=True)
print(kneedle.knee_y)


# In[68]:


kneedle.plot_knee()
plt.show()


# In[69]:


from sklearn.cluster import DBSCAN


# In[70]:


eps=25900
min_samples=6


# In[71]:


db = DBSCAN(eps=eps, min_samples =min_samples)
clusters = db.fit_predict(data)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_


# In[72]:


n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
from sklearn import metrics
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(data, labels))


# In[73]:


from sklearn import metrics
fm=metrics.fowlkes_mallows_score(data['DEATH_EVENT'].astype(int),
                            labels.astype(int))
print("Fowlkes-Mallows Score:",fm)


# In[74]:


count0 = list(labels).count(0)
count1 = list(labels).count(1)
print("The number of 1s (High Death risk):" ,count1)
print("The number of 0s (Low Death Risk):" ,count0)


# In[75]:


LABEL_COLOR_MAP = {0 : 'r',1 : 'b', -1:'k',2: 'y'}
label_color = [LABEL_COLOR_MAP[l] for l in labels]


# In[76]:


fig = plt.figure() 
ax = Axes3D(fig) 
ax.scatter(data.iloc[:, 0].astype(float), data.iloc[:, 6].astype(float), data.iloc[:,10].astype(float),c=label_color) 
plt.show()


# In[77]:


plt.scatter(data.iloc[:,4].astype(float),data.iloc[:,6].astype(float),c=label_color) 
plt.show()


# In[78]:


plt.scatter(data.iloc[:,2].astype(float),data.iloc[:,6].astype(float),c=label_color) 
plt.show()


# In[ ]:




