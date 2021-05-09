import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

traindata = pd.DataFrame(pd.read_csv("bio_train.dat", "\t"))
traindata = traindata.sample(n=20000, replace=False)
dataset = np.array(traindata)
print(dataset.shape)

# create dataset pickle
dbfile = open('kdd_reduced_20k.pickle', 'wb') 
pickle.dump(dataset, dbfile)                      
dbfile.close()
print(dataset)

# load the dataset pickle
dbfile = open("kdd_reduced_20k.pickle", "rb" )
favorite_color = pickle.load(dbfile)
print(favorite_color)

# mat = dataset[:10, :]
# u, s, v = np.linalg.svd(mat)
# u = u[:, :2]
# print(u)
# print(u[2, :])
# ans = np.linalg.norm(u[2, :])**2
# print(ans)