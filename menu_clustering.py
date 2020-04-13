#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv('menu2.csv')
df2 = df[['name','protein','fat']]
#print(df2)
f_name = df2['name'].values
f_name = np.array(list(zip(f_name)))
f1 = df2['protein'].values
f1_x = np.array(list(zip(f1)))

f2 = df2['fat'].values
f2_x = np.array(list(zip(f2)))

Q3_f1, Q1_f1 = np.percentile(f1_x, [75,25])
Q3_f2, Q1_f2 = np.percentile(f2_x, [75,25])

iqr_f1 = Q3_f1 - Q1_f1
iqr_f2 = Q3_f2 - Q1_f2

outlier_f1 = Q3_f1 + 1.5*iqr_f1
outlier_f2 = Q3_f2 + 1.5*iqr_f2

id = []
name= []

len_f1 = len(f1_x)
for i in range(len_f1):
    id.append(i+1)
    name.append([i+1,f_name[i]])
    if f1_x[i] > outlier_f1:
        f1_x[i] = np.nan        
    if f2_x[i] > outlier_f2:
        f2_x[i] = np.nan
X = np.asarray(list(zip(id, f1_x, f2_x)))
df2 = pd.DataFrame(X, columns=['id','protein','fat'])
df2= df2.dropna(axis=0)
f1 = df2['protein'].values
f2 = df2['fat'].values
X = np.array(list(zip(f1, f2)))
k = 4
#X 무작위 중심 좌표
C_x = np.random.randint(0,np.max(X)-20, size=k)
C_y = np.random.randint(0,np.max(X)-20, size=k)
C = np.array(list(zip(C_x,C_y)), dtype=np.float32)
plt.scatter(f1,f2, c='#050505', s =20)
plt.xlabel("Protein Feature")
plt.ylabel("fat Feature")
plt.title("Protein fat Data")
kmeans = KMeans(n_clusters=k)
kmeans = kmeans.fit(X)
test = kmeans.fit_predict(X)
labels = kmeans.predict(X)
centroids = kmeans.cluster_centers_
fig2 = plt.figure()
kx = fig2.add_subplot(111)
for i in range(k):
    points = np.array([X[j] for j in range(len(X)) if labels[j] == i])
    kx.scatter(points[:,0], points[:,1], s= 20, cmap='rainbow')
kx.scatter(centroids[:,0], centroids[:,1], marker='*',s=200, c='#050505')
plt.xlabel("Protein Feature")
plt.ylabel("fat Feature")
plt.title("Protein and fat cluster")
best = np.asarray(list(zip(id, f1, f2,labels)))


