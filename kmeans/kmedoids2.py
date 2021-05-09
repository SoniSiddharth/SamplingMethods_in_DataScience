import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import linalg as LA
import math
import sklearn
from sklearn_extra.cluster import KMedoids
from sklearn.decomposition import PCA
import pickle

# pip install scikit-learn-extra

# x1 , y1 = np.random.multivariate_normal((0,0),[[1,0],[0,1]],1000).T
# x2 , y2 = np.random.multivariate_normal((1,5),[[1,0],[0,1]],1000).T
# plt.scatter(x1,y1)
# plt.scatter(x2,y2)
# plt.show()
#  -----------------------------------------------------------------------------------------------------------
"""# Artificial Dataset"""

#create/get the dataset
# a_dataset = []
# d = 2 #number of dimensions 
# k0 = 10 #actual number of clusters
# sz = 10000

#tuple = center of gaussian = (0,0,0,..)
# t = []
# for i in range(d):
#   t.append(0)
# t = tuple(t)
# print(t)

#identity covarience matrix
# cov = [[0 for i in range(d)] for j in range(d)]
# for i in range(d):
#   cov[i][i] = 1 

# print(cov)

# (i,j) = (0,0)
# for l in range(k0):
#   r = np.random.randint(-1000,1000,size=1)[0]
#   x, y = np.random.multivariate_normal((i,j),cov,int(sz/k0)+r).T
#   plt.scatter(x,y)
#   print(int(sz/k0)+r)
#   for p in range(int(sz/k0)+r):
#     [x1,y1] = [x[p],y[p]]
#     a_dataset.append([x1,y1])
#   if l < 10:
#     i+=8
#     j+=16
#   else :
#     i+=16
#     j-=8
# plt.show()

# ------------------------------------------------------------------------------------------

"""# K-Means generalized

## Function to get nearest center to a datapoint
"""

def nearest(pt,Q):
  min_dist_sq = math.inf
  closest_center = 0
  for c in Q :
    c = np.array(c)
    pt - np.array(pt)
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
    print("inside coreset func")
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
    
    print("done prob distribution")

    #sample m points from this distribution       
    a = [i for i in range(len(dataset))]
    sample = np.random.choice(a,size = m, replace = False ,p=w )

    print("sampled")

    coreset = {}
    for indx in sample:
      p = dataset[indx]
      coreset[tuple(p)] = 1/(m*q[tuple(p)]) #point and weight

    print('coreset done')
    
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

"""CLustering on Coreset and Comparison with optimal Kmeans solution"""

# data2 = []
# wt2 = coreset #since wt is a dictionary with key = pt and value as its wt and we created coreset to store the same thing
# for p in coreset :
#   data2.append([p[0],p[1]])

# centers2 = kmeans_plusplus(data2,10)
# Q_c = kmeans(centers2, data2,wt2)

# cost2, dic = kmeans_cost(Q_c,dataset,wt)

# print(optimal_cost)
# print(cost2)
# error = (abs(cost2-optimal_cost)/optimal_cost)*100
# print(error)



"""# K-Medoids

## init_medoids(k,dataset)
"""

#initialize random k medoids

#what is a medoid ?
# Medoids are representative objects of a data set or a cluster within a data 
# set whose average dissimilarity to all the objects in the cluster is minimal. 
# Medoids are similar in concept to means or centroids, but medoids are always restricted to be members of the data set.
def init_medoids(k,dataset):
  arr = [i for i in range(len(dataset))]
  rndm_indices = np.random.choice(arr,size=k,replace=False)
  medoids = []
  for i in rndm_indices:
    medoids.append(dataset[i])
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

  q = 0

  for oj in dataset :
    q += 1
    if q ==100:
      q=0
    #if oj in medoids:
      #continue
    score = math.inf 
    closest = 0
    for oi in medoids:
      temp = manhattan(oj,oi)
      if temp < score : #meaning this is the closest medoid so far
        score = temp
        closest = oi
    cost += wt[tuple(oj)]*score
    dic[tuple(closest)].append(oj) #assign the non-medoid oj to its closest medoid 
  print("costn done")
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
    nsize = len(dataset)
    while True :
      #assign all points to their nearest medoids and calculate the cost of clustering

      dic,cost = costnassign(medoids,dataset,wt)  
      print(cost)
      #this cost must be = cost(prev) - cost_change
      # show(dic,dataset)

      #find the best medoid, non-medoid pair to swap (that medoid becomes non-mediod and the non-medoid becomes a new medoid)
      oi_best, oh_best = 0,0
      cost_change_min = math.inf
      medl = len(medoids)
      print(medl)
      print(nsize)
      for i in range(medl) :
        oi = medoids[i]
        # print("h2")
        for h in range(nsize) :
          # print("h1")
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
      print("cost change is --> ", cost_change_min)
      if cost_change_min < 0:
        medoids.remove(oi_best)
        medoids.append(oh_best)
      else :
        break #escape the while loop since the cost did not decrease 

    return medoids

"""# KDD"""

# from google.colab import drive
# drive.mount('/content/drive')

# traindata = pd.DataFrame(pd.read_csv("./kdd_train.dat", '\t'))
# print(traindata.shape)
# print(traindata)

# cols = []
# for j in range(traindata.shape[1]): #change the column names 
#   cols.append(j)
# traindata.columns = cols
# print(traindata.shape)
# print(traindata)

# traindata = traindata.drop(columns=2,axis=1)

dbfile = open("kdd_reduced_1k.pickle", "rb" )
traindata = pickle.load(dbfile)

pca = PCA(n_components=2)
dataset = pca.fit_transform(traindata)

traindata = dataset
kdd = np.array(traindata)

"""# Testing on a Dataset"""

dataset = kdd
arr = dataset
k = 50

#run kmedoids on entire dataset

#sklearn implementation
# print("kmedoids on entire dataset")
# print("shape: {}".format(arr.shape))
# km1 = KMedoids(n_clusters=k,random_state=0,method='alternate',metric='manhattan').fit(np.array(dataset))

# dic1,cost1 = costnassign(km1.cluster_centers_,dataset,None)
# print('entire dataset cost: {}'.format(cost1))


"""## Testing Coresets"""

f = open("myfile2.txt", "a")


coreset_size = [100, 50]

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
        print("km2 done")
        km2 = kmedoids(km2,dset2,wts)
        print("medoid done")
        dic2,cost2 = costnassign(km2,dataset.tolist(),None)
        print("hello")
        print(cost2)
        avg_cost += cost2

    avg_cost = avg_cost/5

    # print("optimal cost is --> ", cost1)
    f.write(ssize)
    f.write(avg_cost)
    print("coreset Length", ssize)
    print("sampling cost is --> ", avg_cost)
    reduction = ((dataset.shape[0] - ssize)/dataset.shape[0])*100
    # error = (abs(avg_cost - cost1)/cost1)*100
    print("reduction in dataset is --> ", reduction)
    f.write(reduction)
    # print("error in clustering cost --> ", error)

f.close()