import scipy.io
import numpy as np
import pandas as pd
from copy import deepcopy
from scipy import random, linalg, dot, diag, all, allclose
import timeit
from scipy.sparse.linalg import svds
from numpy import linalg as LA
import pickle


def Estimate_Leverage_Scores(k,Q):
    # print(1/(k)*np.diag(np.dot(Q,Q.T)))
    print(Q.shape)
    print("finding the array")
    return 1/(k)*np.diag(np.dot(Q.T,Q))

def find_top_vals(lvs_array,A,p):
    temp_list = list(reversed(np.argsort(lvs_array)))
    print(len(temp_list))
    sampled_indices_ = temp_list[0:p]
    print(A[:,temp_list[0]].shape)
    return A[:,sampled_indices_]

def deterministic_leveragescores_sampling(X_matrix,k,V,number_of_entries_at_output):

    d = np.shape(X_matrix)[1]
    
    # V = np.transpose(V[:k,:])
    # print(V.shape)
    print("v ka shape :",V.shape)

    lvs_array = Estimate_Leverage_Scores(k,V)
    print("finding top k")
    A_S = find_top_vals(lvs_array,X_matrix,number_of_entries_at_output)

    return A_S

def calculate_right_eigenvectors_k_svd(X_,k):
    _,_,U = svds(X_,k,return_singular_vectors='vh') 
    print("U ka shape",U.shape)
    return U


traindata = pd.DataFrame(pd.read_csv("bio_train.dat", '\t'))
dataset = np.array(traindata)
R = dataset
X_matrix = R
X_matrix = X_matrix[:,:]
X_matrix = X_matrix.T

d = np.shape(X_matrix)[1]
N = np.shape(X_matrix)[0] -1
print("print the dimenstions")
print("d:",d)
print("N:",N)

dimension_for_k_approximation = 76
number_of_entries_at_output = 1000
_,D,V = np.linalg.svd(X_matrix, full_matrices=False)

# v_k = calculate_right_eigenvectors_k_svd(X_matrix,dimension_for_k_approximation)
print(V.shape)
lvs_array = deterministic_leveragescores_sampling(X_matrix,dimension_for_k_approximation,V,number_of_entries_at_output)
print(lvs_array.shape)