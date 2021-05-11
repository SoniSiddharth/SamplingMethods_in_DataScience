# -*- coding: utf-8 -*-
"""Worm_uniform.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10nImJYmWFsFIJ2tUWZUoWsBwYE8JKx7-
"""

import numpy as np
import pandas as pd
from numpy import linalg as LA
import matplotlib.pyplot as plt


"""Real Dataset"""

traindata = pd.DataFrame(pd.read_csv("../worms/worms_64d.txt", " "))
print(traindata.shape)
# print(traindata)
dataset = np.array(traindata)

"""k-Means generalized

"""

def nearest(pt,Q):
  min_dist_sq = 10**18
  closest_center = 0
  for c in Q :
    c = np.array(c)
    pt = np.array(pt)
    dist_sq = (LA.norm(c-pt))**2
    # print(dist_sq)
    if dist_sq < min_dist_sq : 
      min_dist_sq = dist_sq
      closest_center = c
  # print(closest_center)
  return closest_center, min_dist_sq

def kmeans_cost(Q,dataset,wt):
  #kmeans cost for weighted dataset = summasion w(p)*d(p) ,(w(p)= wt of point p, d(p)= dist of p from its nearest center) 
  cost = 0
  dic = {}
  for c in Q:
    dic[tuple(c)] = []
  for p in dataset :
    c, dp = nearest(p,Q)
    cost += dp
    dic[tuple(c)].append(p)

  return cost,dic #dic stores key as center and value as list of points mapped to that canter

wt = {}
for pt in dataset:
  wt[tuple(pt)] = 1

from sklearn.cluster import KMeans
cluster_model = KMeans(n_clusters=25, init='k-means++', random_state=0).fit(dataset)
centers = cluster_model.cluster_centers_
mod_centers = []
for j in centers:
  mod_centers.append(tuple(j))
# print(len(mod_centers))

optimal_cost,dic = kmeans_cost(mod_centers,dataset,wt)
print("Optimal Cost --> ", optimal_cost)

"""Light Weight Coresets"""
from sklearn.cluster import KMeans
sample_sizes = [35000, 30000, 25000, 20000, 15000, 10000, 6000]

for ssize in sample_sizes:
    uniform_data = traindata.sample(n=ssize, replace=False)
    uniform_data = np.array(uniform_data)

    avg_cost = 0
    """CLustering on Coreset and Comparison with optimal Kmeans solution"""
    for j in range(5):
        cluster_model = KMeans(n_clusters=25, init='k-means++', random_state=0).fit(uniform_data)
        centers = cluster_model.cluster_centers_
        mod_centers = []
        for j in centers:
            mod_centers.append(tuple(j))
        # print(len(mod_centers))

        cost2, dic = kmeans_cost(mod_centers,dataset, wt)
        avg_cost += cost2
    
    avg_cost = avg_cost/5

    print("Uniform sample size --> ", ssize)
    print("Sampling Cost --> ", avg_cost)
    reduction = ((dataset.shape[0]-ssize)/dataset.shape[0])*100
    error = (abs(avg_cost - optimal_cost)/optimal_cost)*100
    print("reduction in dataset is --> ", reduction)
    print("error in clustering cost --> ", error)