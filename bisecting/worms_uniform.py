import numpy as np
import pandas as pd
from numpy import linalg as LA
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans

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

wt = {}
for pt in dataset:
	wt[tuple(pt)] = 1

centers = 25
itr=1
# Q = bisecting_k_means(dataset, centers ,itr, wt)
# mod_centers = Q.keys()
# optimal_cost,dic = kmeans_cost(mod_centers,dataset,wt)

wt = {}
for pt in dataset:
	wt[tuple(pt)] = 1

optimal_cost = 14484705856919.4
print("Optimal Cost --> ", optimal_cost)

"""Light Weight Coresets"""
from sklearn.cluster import KMeans
sample_sizes = [35000, 30000, 25000, 20000, 15000, 10000, 6000]

for ssize in sample_sizes:
	uniform_data = traindata.sample(n=ssize, replace=False)
	uniform_data = np.array(uniform_data)

	avg_cost = 0
	"""CLustering on Coreset and Comparison with optimal Kmeans solution"""
	for j in range(3):
		Q = bisecting_k_means(uniform_data, centers , 1, wt)
		mod_centers = Q.keys()
		cost2, dic = kmeans_cost(mod_centers, dataset, wt)
		print(cost2)
		avg_cost += cost2
	
	avg_cost = avg_cost/3

	print("Uniform sample size --> ", ssize)
	print("Sampling Cost --> ", avg_cost)
	reduction = ((dataset.shape[0]-ssize)/dataset.shape[0])*100
	error = (abs(avg_cost - optimal_cost)/optimal_cost)*100
	print("reduction in dataset is --> ", reduction)
	print("error in clustering cost --> ", error)