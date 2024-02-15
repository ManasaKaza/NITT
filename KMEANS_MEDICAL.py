#!/usr/bin/env python
# coding: utf-8

# In[37]:


import numpy as np
import pandas as pd
import numpy.matlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[38]:

data = pd.read_csv('medical.csv')
data.head(5)


# In[39]:


data.nunique()


# In[40]:


data.isnull().sum()


# In[41]:


cluster_data = data[['age','DEATH_EVENT']].copy(deep=True)
cluster_data.dropna(axis=0, inplace=True)
cluster_data.sort_values(by=['age','DEATH_EVENT'], inplace=True)
cluster_array = np.array(cluster_data)
cluster_array


# In[42]:


def calc_distance(X1, X2):
    return (sum((X1 - X2)**2))**0.5


# In[43]:


def assign_clusters(centroids, cluster_array):
    clusters = []
    for i in range(cluster_array.shape[0]):
        distances = []
        for centroid in centroids:
            distances.append(calc_distance(centroid, 
                                           cluster_array[i]))
        cluster = [z for z, val in enumerate(distances) if val==min(distances)]
        clusters.append(cluster[0])
    return clusters


# In[44]:


def calc_centroids(clusters, cluster_array):
    new_centroids = []
    cluster_df = pd.concat([pd.DataFrame(cluster_array),
                            pd.DataFrame(clusters, 
                                         columns=['cluster'])], 
                           axis=1)
    for c in set(cluster_df['cluster']):
        current_cluster = cluster_df[cluster_df['cluster']                                     ==c][cluster_df.columns[:-1]]
        cluster_mean = current_cluster.mean(axis=0)
        new_centroids.append(cluster_mean)
    return new_centroids


# In[45]:


def calc_centroid_variance(clusters, cluster_array):
    sum_squares = []
    cluster_df = pd.concat([pd.DataFrame(cluster_array),
                            pd.DataFrame(clusters, 
                                         columns=['cluster'])], 
                           axis=1)
    for c in set(cluster_df['cluster']):
        current_cluster = cluster_df[cluster_df['cluster']                                     ==c][cluster_df.columns[:-1]]
        cluster_mean = current_cluster.mean(axis=0)
        mean_repmat = np.matlib.repmat(cluster_mean, 
                                       current_cluster.shape[0],1)
        sum_squares.append(np.sum(np.sum((current_cluster - mean_repmat)**2)))
    return sum_squares


# In[46]:


print(cluster_array[100:])


# In[47]:


k = 15
cluster_vars = []

centroids = [cluster_array[i+2] for i in range(k)]
clusters = assign_clusters(centroids, cluster_array)
initial_clusters = clusters
print(0, round(np.mean(calc_centroid_variance(clusters, cluster_array))))

for i in range(10):
    centroids = calc_centroids(clusters, cluster_array)
    clusters = assign_clusters(centroids, cluster_array)
    cluster_var = np.mean(calc_centroid_variance(clusters, 
                                                 cluster_array))
    cluster_vars.append(cluster_var)
    print(i+1, round(cluster_var))


# In[48]:


plt.subplots(figsize=(9,6))
plt.plot(cluster_vars)
plt.xlabel('Iterations')
plt.ylabel('Mean Sum of Squared Deviations');
plt.savefig('mean_ssd', bpi=150)


# In[49]:


plt.subplots(figsize=(9,6))
plt.scatter(x=cluster_array[:,0], y=cluster_array[:,1], 
            c=clusters, cmap=plt.cm.Spectral);
plt.xlabel('age')
plt.ylabel('DEATH_EVENT');
plt.savefig('final_clusters', bpi=150)


# In[50]:



from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# In[51]:


from sklearn.cluster import KMeans
inertia = []
for i in range(1,15):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(data)
    inertia.append((i,kmeans.inertia_,))
    
plt.plot([w[0] for w in inertia],[w[1] for w in inertia], marker="X")
print(inertia)


# In[52]:


from yellowbrick.cluster import KElbowVisualizer
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,12)).fit(data)
visualizer.show()


# In[53]:


clusters = 5

kmeans = KMeans(n_clusters=clusters)
kmeans = kmeans.fit(data)
labels = kmeans.predict(data)
C_center = kmeans.cluster_centers_
print(labels,"\n",C_center)


# In[54]:


dfGroup = pd.concat([data,pd.DataFrame(labels, columns= ['Group'])], axis=1, join='inner')
dfGroup.head()


# In[55]:


dfGroup.groupby("Group").aggregate("mean").plot.bar()


# In[56]:


clustering_kmeans = KMeans(n_clusters = 5)
data['clusters'] = clustering_kmeans.fit_predict(data)


# In[57]:


reduced_data = PCA(n_components=2).fit_transform(data)
results = pd.DataFrame(reduced_data,columns=['pca1','pca2'])

sns.scatterplot(x="pca1", y="pca2", hue=data['clusters'], data=results)
plt.show()


# In[58]:


sns.scatterplot(x=data['DEATH_EVENT'], y=data['age'], hue=data['DEATH_EVENT'], data=results)
plt.show()


# In[59]:


from sklearn import metrics
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(data, labels))


# In[60]:


from sklearn import metrics
fm=metrics.fowlkes_mallows_score(data['DEATH_EVENT'].astype(int),
                            labels.astype(int))
print("Fowlkes-Mallows Score:",fm)


# In[ ]:




