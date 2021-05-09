import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.linalg as LA
import pickle
import math
from sklearn.cluster import KMeans

dbfile = open("../kdd_reduced.pickle", "rb" )
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

def kmeans_centers(k,dataset,wt):
	cluster_model = KMeans(n_clusters=k, init='k-means++', random_state=0).fit(dataset,sample_weight=wt)
	centers = cluster_model.cluster_centers_
	mod_centers = []
	for j in centers:
		mod_centers.append(tuple(j))
	return mod_centers

def cluster_cost(center,cluster_data,wt):
	cost=0
	center=np.array(center)
	for pt in cluster_data:
		pt=np.array(pt)
		dist_sq = (LA.norm(center-pt))**2
		cost+=dist_sq*wt[tuple(pt)]
	return cost

def bisecting_k_means(dataset,k,itr,wt):
	Q={}
	minSSE=math.inf
	a=[]
	for i in dataset:
		a.append(wt[tuple(i)])
	for i in range(itr):
		q=kmeans_centers(2,dataset,a)
		SSE,clusters=kmeans_cost(q,dataset,a)
		if SSE<minSSE:
			minSSE=SSE
			temp_clusters=clusters

	for i in temp_clusters:
		Q[i]=temp_clusters[i]

	while(len(Q)<k):
		maxSSE=-1
		#choose cluster from the set of clusters Q which has maximum SSE    
		for i in Q:
			SSE=cluster_cost(i,Q[i],wt)
			if SSE>maxSSE:
				maxSSE=SSE
				center_with_maxSSE=i
				cluster_with_maxSSE=Q[i]
		#deleting the center/cluser with maxSSE 
		del Q[center_with_maxSSE]
		#Split that clusters into two and choose the pair with minimun SSE (from some itr itrations over the chosen maxSSE cluster)
		minSSE=math.inf
		a=[]
		for j in cluster_with_maxSSE:
			a.append(wt[tuple(j)])
		for i in range(itr):
			q=kmeans_centers(2,cluster_with_maxSSE,a)
			SSE,clusters=kmeans_cost(q,cluster_with_maxSSE,wt)
			if SSE<minSSE:
				minSSE=SSE
				clusters_with_minSSE=clusters

		for i in clusters_with_minSSE:
			Q[i]=clusters_with_minSSE[i]
	print("bisecting finish")
	return Q

def leverage_sampling(data, red_size):
	print("svd started")
	u, s, v = np.linalg.svd(data)
	print("svd done")
	u = u[:, :77]
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

centers = 50
itr=1
Q = bisecting_k_means(dataset, centers ,itr, wt)
mod_centers = Q.keys()
optimal_cost,dic = kmeans_cost(mod_centers,dataset,wt)

coreset_size = [21000, 18000, 15000, 12000, 9000, 6000, 3600]
coreset = leverage_sampling(dataset, 25000)
print("sampling created")
coreset = np.array(coreset)

for ssize in coreset_size:
	condensed_set = coreset[:ssize, :]
	print(condensed_set.shape)
	avg_cost = 0
	for itr in range(3):
		Q = bisecting_k_means(condensed_set, centers , 1, wt)
		mod_centers = Q.keys()
		cost2, dic = kmeans_cost(mod_centers, dataset, wt)
		avg_cost += cost2
	avg_cost = avg_cost/3

	print("optimal cost is --> ", optimal_cost)
	print("length of coreset --> ", ssize)
	print("sampling cost is --> ", avg_cost)
	reduction = ((dataset.shape[0] - ssize)/dataset.shape[0])*100
	error = (abs(avg_cost - optimal_cost)/optimal_cost)*100
	print("reduction in dataset is --> ", reduction)
	print("error in clustering cost --> ", error)