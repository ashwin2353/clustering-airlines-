# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 22:07:39 2022

@author: ashwi
"""
import pandas as pd
df = pd.read_excel("EastWestAirlines.xlsx",1)
df.shape
df.head()
df.dtypes

df.isnull().sum()

# duplicate columns and rows
df.duplicated()
df[df.duplicated()] # hence no duplicates between the rows
len(df[df.duplicated()])

df.columns.duplicated() # heance there is no duplicates between the columns
(df.columns.duplicated()).sum()

# we don't need ID variable so we are removing the variable
df.drop(columns=["ID#"],axis=1,inplace=True)
df.dtypes
df.shape
#===================================================================
# Standaization the data
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
Y = SS.fit_transform(df)
X = pd.DataFrame(Y)
#===================================================================
#####################     K-Means Clusterig    #########################
from sklearn.cluster import KMeans
km = KMeans()

inertia =[]
for i in range(1,11):
    km = KMeans(n_clusters=i, random_state=0)
    km.fit(X)
    inertia.append(km.inertia_)

print(inertia)

# Elbow plot
%matplotlib qt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

plt.plot(range(1,11),inertia)
plt.title("Elbow Method")
plt.xlabel("no.cluster")
plt.ylabel("inertia")
plt.show()

# scree plot
import seaborn as sns
d1 = {"kvalue":range(1,11),"inertiavalues":inertia}
d2 = pd.DataFrame(d1)
sns.barplot(x="kvalue",y="inertiavalues",data= d2,)

#Therefore the variance in inertia  b/w 4th and 5th cluster is less we can go with 4 clusters
# so we can use 4 and 5 cluster but i choose 4 clusters

km = KMeans(n_clusters=4, n_init=30)
km.fit(X)
Y1 = km.predict(X)
X1 = pd.DataFrame(Y1,columns=["cluster"])
X1.value_counts()

new_data = pd.concat([df,X1],axis=1)

#===============================================================================
############################  Hierarchical clustering  ##############################

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=4, affinity="euclidean",linkage="complete")
Y = cluster.fit_predict(X)
Y = pd.DataFrame(Y, columns=["cluster1"])
Y.value_counts()
new_data = pd.concat([df,Y],axis=1)

#===================================================================================
##############################    DBSCAN   ##########################################
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=1,min_samples=3)
dbscan.fit_predict(X)
Y = dbscan.labels_
Y = pd.DataFrame(Y,columns=["cluster2"])
Y.value_counts()
clustered = pd.concat([df,Y],axis=1)
noise_data = clustered[clustered["cluster2"]==-1]
noise_data 
final_data = clustered[clustered["cluster2"]==0]
final_data 

# outliers are removed from the final_data and we can use this final_data for other clustering techniques for better resluts 

# therefore K-Means are prividing better results



















