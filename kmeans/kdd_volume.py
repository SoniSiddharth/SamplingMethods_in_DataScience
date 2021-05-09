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

traindata = pd.DataFrame(pd.read_csv("bio_train.dat", "\t"))
# traindata = traindata.sample(n=1000, replace=False)
dataset = np.array(traindata)
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

def projection(A,S):
	mat = []
	for j in S:
		mat.append(list(j))
	u, s, v = np.linalg.svd(mat)
	print("svd done")
	u = u[:,:77]
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

from sklearn.cluster import KMeans
# cluster_model = KMeans(n_clusters=25, init='k-means++', random_state=0).fit(dataset)
# centers = cluster_model.cluster_centers_
# mod_centers = []
# for j in centers:
# 	mod_centers.append(tuple(j))

# optimal_cost,dic = kmeans_cost(mod_centers,dataset,wt)

optimal_cost = 1541240860988.87

coreset_size = [38000, 32000, 27000, 22000, 16000, 10000, 6000]
# coreset_size = [400, 350, 300, 250, 200, 150]

for ssize in coreset_size:
	t = 2
	s = ssize//t
	vol_sam = volume_samp(dataset,t,s)
	coreset = []
	for j in vol_sam:
		coreset.append(list(j))
	dsize = len(coreset)
	print(dsize)
	avg_cost = 0
	for itr in range(5):
		cluster_model = KMeans(n_clusters=50, init='k-means++', random_state=0).fit(coreset)
		centers = cluster_model.cluster_centers_
		mod_centers = []
		for j in centers:
			mod_centers.append(tuple(j))
		# print(len(mod_centers))
		cost2, dic = kmeans_cost(mod_centers,dataset,wt)
		avg_cost += cost2
	avg_cost = avg_cost/5

	print("optimal cost is --> ", optimal_cost)
	print("length of coreset --> ", dsize)
	print("sampling cost is --> ", avg_cost)
	reduction = ((dataset.shape[0] - dsize)/dataset.shape[0])*100
	error = (abs(avg_cost - optimal_cost)/optimal_cost)*100
	print("reduction in dataset is --> ", reduction)
	print("error in clustering cost --> ", error)