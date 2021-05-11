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

"""Light Weight Coresets"""

def light_weight_coreset(dataset,m):
	#calculate the mean of all data points
	mu = [0]*(len(dataset[0]))
	for p in dataset :
	  for k in range(len(p)):
		mu[k] += p[k]
	for k in range(len(p)):
	  mu[k] = mu[k]/len(dataset)

	#first term in prob distribution
	a = 1/(2*len(dataset))

	#denominator in second term of prob distribution
	sum_dsq = 0
	mu = np.array(mu)
	for p in dataset :
	  p = np.array(p)
	  sum_dsq += (LA.norm(mu-p))**2

	#assign probability to each point
	q = {}
	w = []
	for p in dataset :
	  p = np.array(p)
	  dsq = (LA.norm(mu-p))**2
	  q[tuple(p)] = a + (1/2)*(dsq/sum_dsq)
	  w.append(q[tuple(p)])
	

	#sample m points from this distribution       
	a = [i for i in range(len(dataset))]
	sample = np.random.choice(a,size = m, replace = False ,p=w )

	coreset = {}
	for indx in sample:
	  p = dataset[indx]
	  coreset[tuple(p)] = 1/(m*q[tuple(p)]) #point and weight

	return coreset

coreset_size = [35000, 30000, 25000, 20000, 15000, 10000, 6000]
# coreset_size = [14000, 12000, 10000, 8000, 6000, 4000, 2400]
from sklearn.cluster import KMeans

for ssize in coreset_size:
  uniform_data = traindata.sample(n=ssize, replace=False)
  uniform_data = np.array(uniform_data)
  avg_cost = 0
  for j in range(3):
	mod_centers = k_center(uniform_data, centers, wt)
	cost2 = kcenter_cost(mod_centers, dataset)
	avg_cost += cost2
  avg_cost = avg_cost/3

  print("coreset Length", ssize)
  print("sampling cost is --> ", avg_cost)
  reduction = ((dataset.shape[0] - ssize)/dataset.shape[0])*100
  error = (abs(avg_cost - optimal_cost)/optimal_cost)*100
  print("reduction in dataset is --> ", reduction)
  print("error in clustering cost --> ", error)
