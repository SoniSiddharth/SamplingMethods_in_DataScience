import numpy as np
import pandas as pd
from numpy import linalg as LA
import matplotlib.pyplot as plt
import pickle
import math
from sklearn.decomposition import PCA

"""Real Dataset"""

traindata = pd.DataFrame(pd.read_csv("../worms/worms_64d.txt", " "))
dataset = np.array(traindata)

# dbfile = open("../worms_reduced_30k.pickle", "rb" )
# dataset = pickle.load(dbfile)

# pca = PCA(n_components=5)
# dataset = pca.fit_transform(dataset)
print(dataset.shape)

def nearest(pt,Q):
  min_dist_sq = 10**18
  closest_center = 0
  for c in Q :
    c = np.array(c)
    pt = np.array(pt)
    dist_sq = (LA.norm(c-pt))
    # print(dist_sq)
    if dist_sq < min_dist_sq:
      min_dist_sq = dist_sq
      closest_center = c
  # print(closest_center)
  return closest_center, min_dist_sq

def kcenter_cost(centers, data):
    cost = 0
    dic = {}
    for c in centers:
        dic[c] = []
    for p in data:
        ctr, dp = nearest(p,centers)
        dic[tuple(ctr)].append(dp)
    for j in dic:
        x = max(dic[j])
        if x>cost:
            cost = x
    return cost 

import random
def k_center(dataset,k,wt):
  Q=[]
  m=len(dataset)
  c1=random.randint(0,m)
  Q.append(tuple(dataset[c1]))
  while(len(Q)<k):
    maximum_dist_point=Q[0]
    maximum_dist=-1
    for pt in dataset:
      if tuple(pt) not in Q:
        center_dist={}
        for c in Q:
          c=np.array(c)
          pt=np.array(pt)          
          dist_sq = (LA.norm(c-pt))**2
          center_dist[tuple(c)]=dist_sq
        min_key = min(center_dist, key=center_dist.get)
        max_dist=center_dist[min_key]
        if max_dist>maximum_dist:
          maximum_dist_point=pt
          maximum_dist=max_dist
    Q.append(tuple(maximum_dist_point))
  return Q

wt = {}
for pt in dataset:
  wt[tuple(pt)] = 1

from sklearn.cluster import KMeans
centers = 25
mod_centers = k_center(dataset, centers, wt)
optimal_cost = kcenter_cost(mod_centers, dataset)
print("optimal cost is --> ", optimal_cost)


def projection(A,S):
	mat = []
	for j in S:
		mat.append(list(j))
	u, s, v = np.linalg.svd(mat)
	print("svd done")
	u = u[:,:64]
	component = (np.dot(u.T, np.dot(u, A.T))).T
	ans = A - component
	return ans

def union_a_b(S,t):
	for row in t:
		x = tuple(row)
		if x not in S:
			S[x] = 0
	return S

def frob(A):
	sum=0
	for i in range(len(A)):
		for j in range(len(A[i])):
			sum+=pow(A[i][j],2)
	return sum

def frob_row(E):
	#E=A[i]
	sum=0
	for i in range(len(E)):
		sum+=pow(E[i],2)
	return sum

def sum_arr(arr):
	s=0
	for i in range(len(arr)):
		s=s+arr[i]
	return s

def volume_samp(A,t,s):
	E=A
	m,n=np.shape(A)
	S= {}

	P=[]
	P=[0 for i in range(m)]

	for j in range(t):
		den = (np.linalg.norm(E))**2
		T=[]  
		for i in range(m):
			P[i] = (np.linalg.norm(E[i,:])**2)/den

		a=[i for i in range(len(A))] 
		T_index = np.random.choice(a, size = s, replace = False ,p=P )
		T_matrix=A[T_index,:]
		T_matrix=np.array(T_matrix)
		S = union_a_b(S,T_matrix)
		E=projection(A,S)
	return S

x = 0.2
k=int(dataset.shape[0]*x)
t=2
e=0.2
s = int(k/e)

wt = {}
for pt in dataset:
	wt[tuple(pt)] = 1

coreset_size = [35000, 30000, 25000, 20000, 15000, 10000, 6000]
# coreset_size = [14000, 12000, 10000, 8000, 6000, 4000, 2400]
from sklearn.cluster import KMeans

for ssize in coreset_size:
  t = 2
  s = ssize//t
  vol_sam = volume_samp(dataset,t,s)
  coreset = []
  for j in vol_sam:
    coreset.append(list(j))
  dsize = len(coreset)
  print(dsize)
  # running 5 times and taking average
  avg_cost = 0
  for j in range(3):
    mod_centers = k_center(coreset, centers, wt)
    cost2 = kcenter_cost(mod_centers, dataset)
    avg_cost += cost2
  avg_cost = avg_cost/3

  print("coreset Length", dsize)
  print("sampling cost is --> ", avg_cost)
  reduction = ((dataset.shape[0] - dsize)/dataset.shape[0])*100
  error = (abs(avg_cost - optimal_cost)/optimal_cost)*100
  print("reduction in dataset is --> ", reduction)
  print("error in clustering cost --> ", error)
