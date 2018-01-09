from PermutationBasedIndex.distance_metric import pairwise_cosine_distance
from math import ceil, sqrt, floor
import numpy as np
import scipy as sp
import random
from time import time
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import KMeans

def reference_set_selection(X,reference_set_size = 10,ref_sel_threshold = 0.5, metric_vector_distance = pairwise_cosine_distance):
    '''
        Distributed Selection: close reference points are neglected based on a threshold
         
    ''' 

    '''
        randomly selects the first reference point
    '''
#         set_id = sample(range(X.shape[0]),1)
    t0 = time()
    
    set_id = [ceil(X.shape[0]/2)]

    current_id = set_id[0]
    
    first_reference = np.nonzero(metric_vector_distance(X[current_id,:],X) > ref_sel_threshold)[1]

    i = 0        
    while len(set_id) < reference_set_size and i < len(first_reference):
        current_id = first_reference[i]
        i += 1
        current_reference = np.nonzero(metric_vector_distance(X[current_id,:],X[set_id,:]) < ref_sel_threshold)[1]
        
        if len(current_reference) == 0:
            set_id.append(current_id)

        del current_reference
        
    
    return X[set_id,:],time()-t0


'''
    Bauckhage C. Numpy/scipy Recipes for Data Science: k-Medoids Clustering[R]. Technical Report, University of Bonn, 2015.
'''
def kMedoids(X, k = 10, tmax=100):
    
    D = pairwise_cosine_distance(X)
    t0 = time()
    # determine dimensions of distance matrix D
    m, n = D.shape

    if k > n:
        raise Exception('too many medoids')
    # randomly initialize an array of k medoid indices
    M = np.arange(n)
    np.random.shuffle(M)
    M = np.sort(M[:k])

    # create a copy of the array of medoid indices
    Mnew = np.copy(M)

    # initialize a dictionary to represent clusters
    C = {}
    for t in range(tmax):
        # determine clusters, i. e. arrays of data indices
        J = sp.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]
        # update cluster medoids
        for kappa in range(k):
            J = np.mean(D[np.ix_(C[kappa],C[kappa])],axis=1)
            j = np.argmin(J)
            Mnew[kappa] = C[kappa][j]
        np.sort(Mnew)
        # check for convergence
        if np.array_equal(M, Mnew):
            break
        M = np.copy(Mnew)
    else:
        # final update of cluster memberships
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]

   
    # return results
    return D[M],time()-t0



def kmeans(X, parameters = {}):
    
    t0 = time()
    if("distance_metric" in parameters):
        f_distance_metric = parameters["distance_metric"]
    else:
        f_distance_metric = pairwise_cosine_distance
    
        
    #D = f_distance_metric(X).T
        
    
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    
    M = kmeans.cluster_centers_
    
    return np.array(M), time()-t0




