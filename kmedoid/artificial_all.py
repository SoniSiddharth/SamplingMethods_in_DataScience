import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import linalg as LA
import math
import sklearn
from sklearn_extra.cluster import KMedoids
from sklearn.decomposition import PCA


"""# Artificial Dataset"""

#create/get the dataset
a_dataset = []
d = 2 #number of dimensions 
k0 = 20 #actual number of clusters
sz = 500

#tuple = center of gaussian = (0,0,0,..)
t = []
for i in range(d):
  t.append(0)
t = tuple(t)
# print(t)

#identity covarience matrix
cov = [[0 for i in range(d)] for j in range(d)]
for i in range(d):
  cov[i][i] = 1 

# print(cov)

netsz = 0
(i,j) = (0,0)
for l in range(k0):
  r = np.random.randint(-10,100,size=1)[0]
  x, y = np.random.multivariate_normal((i,j),cov,int(sz/k0)+r).T
  netsz += int(sz/k0)+r
  plt.scatter(x,y)
  # print(int(sz/k0)+r)
  for p in range(int(sz/k0)+r):
    [x1,y1] = [x[p],y[p]]
    a_dataset.append([x1,y1])
  if l < 10:
    i+=8
    j+=16
  else :
    i+=16
    j-=8
print(netsz)
plt.show()

def add_to_ds(x,y,sz):
  for p in range(sz):
    [x1,y1] = [x[p],y[p]]
    a1_dataset.append([x1,y1])

from sklearn import datasets

digits = datasets.load_digits(return_X_y=True)

digits[0].shape

a1_dataset = []
x, y = np.random.multivariate_normal((-2,-1),cov,150).T
add_to_ds(x,y,100)
plt.scatter(x,y)
x, y = np.random.multivariate_normal((2,-7),cov,1000).T
add_to_ds(x,y,1000)
plt.scatter(x,y)
x, y = np.random.multivariate_normal((10,10),cov,50).T
add_to_ds(x,y,50)
plt.scatter(x,y)
x, y = np.random.multivariate_normal((4,4),cov,1200).T
add_to_ds(x,y,1000)
plt.scatter(x,y)
x, y = np.random.multivariate_normal((-5,5),cov,750).T
add_to_ds(x,y,500)
plt.scatter(x,y)
x, y = np.random.multivariate_normal((12,2),cov,500).T
add_to_ds(x,y,500)
plt.scatter(x,y)
x, y = np.random.multivariate_normal((18,-2),cov,310).T
add_to_ds(x,y,50)
plt.scatter(x,y)
x, y = np.random.multivariate_normal((10,-5),cov,1000).T
add_to_ds(x,y,1000)
plt.scatter(x,y)
x, y = np.random.multivariate_normal((24,-5),cov,200).T
add_to_ds(x,y,200)
plt.scatter(x,y)
x, y = np.random.multivariate_normal((25,10),cov,400).T
add_to_ds(x,y,400)
plt.scatter(x,y)
x, y = np.random.multivariate_normal((16,8),cov,400).T
add_to_ds(x,y,400)
plt.scatter(x,y)
# x, y = np.random.multivariate_normal((0,0),cov,100).T
# add_to_ds(x,y,100)
# x, y = np.random.multivariate_normal((0,0),cov,100).T
# add_to_ds(x,y,100)
# x, y = np.random.multivariate_normal((0,0),cov,100).T
# add_to_ds(x,y,100)
# x, y = np.random.multivariate_normal((0,0),cov,100).T
# add_to_ds(x,y,100)
# x, y = np.random.multivariate_normal((0,0),cov,100).T
# add_to_ds(x,y,100)
# x, y = np.random.multivariate_normal((0,0),cov,100).T
# add_to_ds(x,y,100)
# x, y = np.random.multivariate_normal((0,0),cov,100).T
# add_to_ds(x,y,100)
plt.show()

"""# K-Means generalized

## Function to get nearest center to a datapoint
"""

def nearest(pt,Q):
  min_dist_sq = math.inf
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

"""## K-Means ++"""

#k centers initialization
def kmeans_plusplus(dataset,k):
  Q = []
  #choose the first center randomly
  index = np.random.randint(0,len(dataset))
  Q.append(dataset[index]) 
  
  for i in range(k-1):
    p = [] #store probability associated with each point
    d = [] #store squared dist of each pt to its nearest center
    sum = 0

    for pt in dataset:
      if pt in Q:
        d.append(0)
        continue

      closest_center , min_dist_sq = nearest(pt,Q)
      d.append(min_dist_sq)
      sum += min_dist_sq
    
    for j in range(len(dataset)):
      p.append(d[j]/sum)
    

    next_center_index = np.random.choice([i for i in range(len(dataset))],p = p) #choose the next center based on probabilities
    Q.append(dataset[next_center_index])
  

  return Q

"""## K-Means"""

def kmeans(Q,dataset,wt,n_itrs):

    for itr in range(n_itrs):

      assignment = {}

      for c in Q :
        assignment[c] = []

      #assign each point to nearest center
      for pt in dataset :
        if tuple(pt) in Q :
          continue

        closest_center , min_dist_sq = nearest(pt,Q)

        assignment[tuple(closest_center)].append(pt)

      #recalculate centers
      Q_new = []
      for c in assignment: #c = center, assignment[c] = pts assigned to c

        num = [0]*len(assignment[c][0]) #[0,0,....,0] d 0s, d = dimensions of a point
        denom = 0
        for x in assignment[c]:
          denom += wt[tuple(x)]
          for m in range(len(num)):
            num[m] += x[m]*wt[tuple(x)]
        c_new = []
        for m in range(len(num)):
          c_new.append(num[m]/denom) #generalized kmeans center update : cnew = (summasion w(x)*x)/summasion w(x)
        Q_new.append(tuple(c_new))
      Q = Q_new

    return Q

"""## K-Means Cost and Cluster Assignment"""

def kmeans_cost(Q,dataset):
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

"""# Light Weight Coresets

## light_weight_coreset(dataset,m)
"""

def light_weight_coreset(dataset,m):
    #calculate the mean over all data points
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

"""## coreset_to_data(coreset)"""

def coreset_to_data(coreset):
  data = []
  wts = coreset #since wt is a dictionary with key = pt and value as its wt and we created coreset to store the same thing
  for p in coreset :
    data.append(list(p))
  
  return data, wts

def visualize(coreset):
  x = []
  y = []
  for p in coreset :
    x.append(p[0])
    y.append(p[1])
  plt.scatter(x,y)
  plt.show()
  return



"""# K-Medoids

## init_medoids(k,dataset)
"""

#initialize random k medoids

#what is a medoid ?
# Medoids are representative objects of a data set or a cluster within a data 
# set whose average dissimilarity to all the objects in the cluster is minimal. 
# Medoids are similar in concept to means or centroids, but medoids are always restricted to be members of the data set.
def init_medoids(k,data):
  arr = [i for i in range(len(data))]
  rndm_indices = np.random.choice(arr,size=k,replace=False)
  medoids = []
  for i in rndm_indices:
    medoids.append(data[i])
  return medoids

def manhattan(a,b):
  dist = 0
  # print(a)
  for i in range(len(a)):
    dist += abs(a[i]-b[i])
  return dist

"""## cost_change()"""

#function to get the new assignment of nonmedoids to nearest medoid after swapping oi and oh
def cost_change(oi,oh,dic,dataset,wt):
  #dic => key=center, value=list

  delta_cost = 0

  for i in range(len(dataset)):
    # print(i)
    oj = dataset[i]
    if tuple(oj) in dic.keys() or oj == oh:
      pass #not continue because we don't want to ignore oi

    if oj in dic[tuple(oi)]:
      d2 = math.inf #d2 = dist of oj to the second nearest medoid
      oj2 = 0 #that second nearest medoid
      #finding 2nd nearest medoid
      for m0 in dic.keys() :
        m = list(m0)
        if m == oi:
          continue
        else:
          temp = manhattan(oj,m)
          if temp < d2:
            d2 = temp
            oj2 = m

      # case1
      d1 = manhattan(oj,oi)
      if manhattan(oj,oh) >= d2 : #if oj belongs to the cluster of oi and it is closer to oj2 than to oh, therefore it will be assigned to oj2 and so the 
        #its contribution to the change in cost will be
        delta_cost += wt[tuple(oj)]*(d2 - d1) #this has to be positive
  
      # case2
      else : #if oj belongs to the cluster of oi and it is closer to oh than to oj2
        delta_cost += wt[tuple(oj)]*(manhattan(oj,oh) - d1)



    else : #oj belongs to some other cluster than that of oi

      ow = 0
      for m in dic.keys():
        if oj in dic[m]:
          ow = m
          break
      # case3
      d1 = manhattan(oj,ow)
      d2 = manhattan(oj,oh)

      if d1 < d2 : #if oj was closer to its current cluster center than to oh, then its assignment wouldn't change
      #and hence its contribution to the change in cost will be 0
        delta_cost += 0
      # case4
      else :
        delta_cost += wt[tuple(oj)]*(d2 - d1) #this has to be negative
  
 
  return delta_cost

"""## costnassign(medoids,dataset,wt)"""

def costnassign(medoids,dataset,wt):
  if wt == None:
    wt = {}
    for x in dataset:
      wt[tuple(x)]=1

  cost = 0
  dic = {} #key=medoid :value=assigned points

  for m in medoids:
    dic[tuple(m)] = [m]

  for oj in dataset :
    # if oj in medoids:
    #   continue
    score = math.inf 
    closest = 0
    for oi in medoids:
      temp = manhattan(oj,oi)
      # print(temp)
      if temp < score : #meaning this is the closest medoid so far
        score = temp
        closest = oi
    cost += wt[tuple(oj)]*score
    dic[tuple(closest)].append(oj) #assign the non-medoid oj to its closest medoid 
  return dic,cost

def show(dic,dataset):
  for med in dic:
    ds = np.array(dic[med])
    x = ds[:,0]
    y = ds[:,1]
    plt.scatter(x,y)
  plt.show()
  return

"""## kmedoids(medoids,dataset,wt)"""

#kmedoids clustering algorithm


def kmedoids(medoids,dataset,wt):

    if wt == None:
      wt = {}
      for x in dataset:
        wt[tuple(x)]=1

    while True :

      #assign all points to their nearest medoids and calculate the cost of clustering

      dic,cost = costnassign(medoids,dataset,wt)  
      print(cost)
      #this cost must be = cost(prev) - cost_change
      # show(dic,dataset)

      #find the best medoid, non-medoid pair to swap (that medoid becomes non-mediod and the non-medoid becomes a new medoid)
      oi_best, oh_best = 0,0
      cost_change_min = math.inf

      for i in range(len(medoids)) :
        oi = medoids[i]
        for h in range(len(dataset)) :
          oh = dataset[h]
          #find the cost of replacing this medoid oi with the non medoid oh
          temp = cost_change(oi,oh,dic,dataset,wt)
          # print(temp)
          if temp < cost_change_min :
            cost_change_min = temp
            oi_best,oh_best = oi,oh
          
      # print(cost_change_min)
      # print('expected new cost')
      # cnew = cost + cost_change_min
      # print(cnew)
      if cost_change_min < 0:
        medoids.remove(oi_best)
        medoids.append(oh_best)
      else :
        break #escape the while loop since the cost did not decrease 

    return medoids



print("----------------------------- Clustering on Entire dataset -----------------------------")
#run kmedoids on entire dataset
dataset = a1_dataset
k = 11
#sklearn implementation
km1 = KMedoids(n_clusters=k,random_state=0,method='pam',metric='manhattan').fit(np.array(dataset))

#our code
# km1 = init_medoids(k,dataset.tolist())
# km1 = kmedoids(km1,dataset.tolist(),None)

print("----------------------------- Clustering on Coresets -----------------------------")

#coreset
# m = int(arr.shape[0]*0.01) #coreset  size (0.1% of original data)
coreset = light_weight_coreset(dataset,100) 
dset2,wts = coreset_to_data(coreset)

#kmedoids on coreset
km2 = init_medoids(k,dset2)
km2 = kmedoids(km2,dset2,wts)

"""## Testing Coresets"""

#cost comparison
dic1,cost1 = costnassign(km1.cluster_centers_,dataset,None)
dic2,cost2 = costnassign(km2,dataset,None)
print('optimal cost =>')
print(cost1)
print('coreset cost =>')
print(cost2)
print("relative error")
err = ((cost2-cost1)/cost1)*100
print(err)

# visualize(coreset)

# visualize(dataset)

#uniform sampling

arr = np.array(dataset)
sz=100
a =[i for i in range(len(dataset))]
a = np.random.choice(a,size=sz,replace=False)
u_dataset = []
for i in a:
  u_dataset.append(dataset[i])

# visualize(u_dataset)

km3 = KMedoids(n_clusters=k,random_state=0,method='pam',metric='manhattan').fit(np.array(u_dataset))

dic3,cost3 = costnassign(km3.cluster_centers_,dataset,None)

cost3

#cost comparison
dic3,cost3 = costnassign(km3.cluster_centers_,dataset,None)
print('optimal cost =>')
print(cost1)
print('uniform sampling cost =>')
print(cost3)
print("relative error")
err = ((cost3-cost1)/cost1)*100
print(err)

len(km3.cluster_centers_)

# running 5 times and taking average
  
coreset = light_weight_coreset(dataset, 200)
dset2,wts = coreset_to_data(coreset)
km2 = init_medoids(k,dset2)
km2 = kmedoids(km2,dset2,wts)
dic2,cost2 = costnassign(km2,dataset,None)
# print('about to print cost2')
print(cost2)

coreset_size = [300, 200, 150, 125, 100, 75, 50]

print("----------------------------- Clustering on Coresets -----------------------------")

for ssize in coreset_size:
    print(ssize)
    """CLustering on Coreset and Comparison with optimal Kmedoids solution"""

    # running 5 times and taking average
    avg_cost = 0
    for j in range(5):
        print(j)
        coreset = light_weight_coreset(dataset, ssize)
        dset2,wts = coreset_to_data(coreset)
        km2 = init_medoids(k,dset2)
        km2 = kmedoids(km2,dset2,wts)
        dic2,cost2 = costnassign(km2,dataset,None)
        # print('about to print cost2')
        # print(cost2)
        avg_cost += cost2

    avg_cost = avg_cost/5

    # print("optimal cost is --> ", cost1)
    print("coreset Length", ssize)
    print("sampling cost is --> ", avg_cost)
    reduction = ((len(dataset) - ssize)/len(dataset))*100
    error = ((avg_cost - cost1)/cost1)*100
    print("reduction in dataset is --> ", reduction)
    print("error in clustering cost --> ", error)

print("----------------------------- Clustering on Uniform sampling -----------------------------")

sample_size = [300, 200, 150, 125, 100, 75, 50]
for ssize in sample_size:
    arr = np.array(dataset)
    a = [i for i in range(len(dataset))]
    a = np.random.choice(a,size=ssize,replace=False)
    u_dataset = []
    for i in a:
        u_dataset.append(dataset[i])

    # running 5 times and taking average
    avg_cost = 0
    for j in range(5):
        print(j)
        km3 = KMedoids(n_clusters=k,random_state=0,method='pam',metric='manhattan').fit(np.array(u_dataset))
        #cost comparison
        dic3,cost3 = costnassign(km3.cluster_centers_,dataset,None)
        avg_cost += cost3

    avg_cost = avg_cost/5

    # print("optimal cost is --> ", cost1)
    print("sample size", ssize)
    print("uniform sampling cost is --> ", avg_cost)
    reduction = ((len(dataset) - ssize)/len(dataset))*100
    error = (abs(avg_cost - cost1)/cost1)*100
    print("reduction in dataset is --> ", reduction)
    print("error in clustering cost --> ", error)

"""## Leverage Score

"""


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

optimal_cost = 8107.910232
coreset_size = [300, 200, 150, 125, 100, 75, 50]

coreset = leverage_sampling(np.array(dataset), 400)
coreset = np.array(coreset)
print("sampling created")

print("----------------------------- Clustering on Leverage Sampling -----------------------------")

for ssize in coreset_size:
    condensed_set = coreset[:ssize, :]
    print(condensed_set.shape)
    km4 = KMedoids(n_clusters=11,random_state=0,method='pam',metric='euclidean').fit(np.array(condensed_set))
    #cost comparison
    dic4,cost4 = costnassign(km4.cluster_centers_,dataset,None)

    print("length of coreset --> ", ssize)
    print("sampling cost is --> ", cost4)
    reduction = ((len(dataset) - ssize)/len(dataset))*100
    error = (abs(cost4 - optimal_cost)/optimal_cost)*100
    print("reduction in dataset is --> ", reduction)
    print("error in clustering cost --> ", error)

temp = []
for j in dataset:
  temp.append(tuple(j))
for j in condensed_set:
  if tuple(j) not in temp:
    print("error")
    break

def projection(A,S):
    mat = []
    for j in S:
        mat.append(list(j))
    u, s, v = np.linalg.svd(mat)
    print("svd done")
    u = u[:,:2]
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
k1=int(len(dataset)*x)
t=2
e=0.2
s = int(k1/e)



coreset_size = [300, 200, 150, 125, 100, 75, 50]
print("----------------------------- Clustering on Volume Sampling -----------------------------")

for ssize in coreset_size:
    t = 2
    s = ssize//t
    vol_sam = volume_samp(np.array(dataset),t,s)
    coreset = []
    for j in vol_sam:
        coreset.append(list(j))
    dsize = len(coreset)
    print(dsize)
    avg_cost = 0
    for itr in range(5):
        km5 = KMedoids(n_clusters=12,random_state=0,method='pam',metric='euclidean').fit(np.array(coreset))
        #cost comparison
        dic5,cost5 = costnassign(km5.cluster_centers_,dataset,None)
        avg_cost += cost5
    avg_cost = avg_cost/5

    print("optimal cost is --> ", optimal_cost)
    print("length of coreset --> ", dsize)
    print("sampling cost is --> ", avg_cost)
    reduction = ((len(dataset) - ssize)/len(dataset))*100
    error = (abs(avg_cost - optimal_cost)/optimal_cost)*100
    print("reduction in dataset is --> ", reduction)
    print("error in clustering cost --> ", error)