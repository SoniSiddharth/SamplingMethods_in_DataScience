import numpy as np
import pandas as pd
from numpy import linalg as LA
import matplotlib.pyplot as plt
import pickle
import math
from sklearn.decomposition import PCA

"""Real Dataset"""

# traindata = pd.DataFrame(pd.read_csv("../worms/worms_64d.txt", " "))
# dataset = np.array(traindata)

dbfile = open("../worms/worms_reduced.pickle", "rb" )
dataset = pickle.load(dbfile)

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


def leverage_sampling(data, red_size):
	print("svd started")
	u, s, v = np.linalg.svd(data)
	print("svd done")
	u = u[:, :64]
	norms = []
	N = data.shape[0]
	for j in range(N):
		norms.append((tuple(data[j]), np.linalg.norm(u[j,:])**2))
	norms_sorted = sorted(norms, key=lambda x: x[1], reverse=True)
	reduced_set = []
	for j in range(red_size):
		reduced_set.append(list(norms_sorted[j][0]))
	return reduced_set


wt = {}
for pt in dataset:
	wt[tuple(pt)] = 1

# coreset_size = [35000, 30000, 25000, 20000, 15000, 10000, 6000]
# coreset_size = [14000, 12000, 10000, 8000, 6000, 4000, 2400]

coreset_size = [21000, 18000, 15000, 12000, 9000, 6000, 3600]
from sklearn.cluster import KMeans

coreset = leverage_sampling(dataset, max(coreset_size))
coreset = np.array(coreset)

for ssize in coreset_size:
  condensed_set = coreset[:ssize, :]
  print(condensed_set.shape)
  # running 5 times and taking average
  avg_cost = 0
  for j in range(3):
    mod_centers = k_center(condensed_set, centers, wt)
    cost2 = kcenter_cost(mod_centers, dataset)
    avg_cost += cost2
  avg_cost = avg_cost/3

  print("coreset Length", ssize)
  print("sampling cost is --> ", avg_cost)
  reduction = ((dataset.shape[0] - ssize)/dataset.shape[0])*100
  error = (abs(avg_cost - optimal_cost)/optimal_cost)*100
  print("reduction in dataset is --> ", reduction)
  print("error in clustering cost --> ", error)
