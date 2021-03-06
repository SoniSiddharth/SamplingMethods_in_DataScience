# -*- coding: utf-8 -*-
"""Worms_volume_sampling.ipynb

Automatically generated by Colaboratory.

Original file is located at
		https://colab.research.google.com/drive/1euLUgc01uPQHSBPOzXhGn91CV5A-tHJy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.linalg as LA
import pickle

dbfile = open("../kdd/kdd_reduced.pickle", "rb" )
dataset = pickle.load(dbfile)
print(dataset.shape)

def nearest(pt,Q):
	min_dist_sq = 10**18
	closest_center = 0
	for c in Q:
		c = np.array(c)
		pt - np.array(pt)
		dist_sq = (LA.norm(c-pt))**2
		# print(dist_sq)
		if dist_sq < min_dist_sq: 
			min_dist_sq = dist_sq
			closest_center = c
	return closest_center, min_dist_sq

def kmeans_cost(Q,dataset,wt):
	#kmeans cost for weighted dataset = summasion w(p)*d(p) ,(w(p)= wt of point p, d(p)= dist of p from its nearest center) 
	cost = 0
	dic = {}
	for c in Q:
		dic[tuple(c)] = []
	for p in dataset:
		c, dp = nearest(p,Q)
		cost += dp
		dic[tuple(c)].append(p)
	return cost, dic


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

from sklearn.cluster import KMeans
cluster_model = KMeans(n_clusters=50, init='k-means++', random_state=0).fit(dataset)
centers = cluster_model.cluster_centers_
mod_centers = []
for j in centers:
	mod_centers.append(tuple(j))

optimal_cost,dic = kmeans_cost(mod_centers,dataset,wt)


coreset_size = [21000, 18000, 15000, 12000, 9000, 6000, 3600]

coreset = leverage_sampling(dataset, 25000)
print("sampling created")
coreset = np.array(coreset)

for ssize in coreset_size:
	condensed_set = coreset[:ssize, :]
	print(condensed_set.shape)
	avg_cost = 0
	for itr in range(5):
		cluster_model = KMeans(n_clusters=50, init='k-means++', random_state=0).fit(condensed_set)
		centers = cluster_model.cluster_centers_
		mod_centers = []
		for j in centers:
			mod_centers.append(tuple(j))
		# print(len(mod_centers))
		cost2, dic = kmeans_cost(mod_centers,dataset,wt)
		avg_cost += cost2
	avg_cost = avg_cost/5

	print("optimal cost is --> ", optimal_cost)
	print("length of coreset --> ", ssize)
	print("sampling cost is --> ", avg_cost)
	reduction = ((dataset.shape[0] - ssize)/dataset.shape[0])*100
	error = (abs(avg_cost - optimal_cost)/optimal_cost)*100
	print("reduction in dataset is --> ", reduction)
	print("error in clustering cost --> ", error)