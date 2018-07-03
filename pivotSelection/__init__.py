from PermutationBasedIndex.distance_metric import pairwise_cosine_distance
from math import ceil, sqrt, floor
import numpy as np
import scipy as sp
import random
from time import time
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import KMeans,Birch
from random import sample

def random_select_pivot(X,parameters = {}):
    '''
        Distributed Selection: close reference points are neglected based on a threshold
         
    ''' 

    '''
        randomly selects the first reference point
    '''
    if("k" in parameters):
        k = parameters["k"]
    else:
        k = 10
    
    t0 = time()
    
    set_id = sample(range(X.shape[0]),k)
    
    return X[set_id,:],time()-t0

def reference_set_selection(X,parameters = {}):
    '''
        Distributed Selection: close reference points are neglected based on a threshold
         
    ''' 

    '''
        randomly selects the first reference point
    '''
    
    if("k" in parameters):
        reference_set_size = parameters["k"]
    else:
        reference_set_size = 25
    if("function_distance" in parameters):
        f_distance_metric = parameters["function_distance"]
    else:
        f_distance_metric = pairwise_cosine_distance
    if("distance_metric" in parameters):
        distance_metric = parameters["distance_metric"]
    else:
        distance_metric = "euclidean"    
    if("ref_sel_threshold" in parameters):
        ref_sel_threshold = parameters["ref_sel_threshold"]
    else:
        ref_sel_threshold = 0.5
    
    t0 = time()
    
    set_id = [ceil(X.shape[0]/2)]

    current_id = set_id[0]
    
    first_reference = np.nonzero(f_distance_metric(X[current_id,:],X,metric_distance=distance_metric) > ref_sel_threshold)[1]

    i = 0        
    while len(set_id) < reference_set_size and i < len(first_reference):
        current_id = first_reference[i]
        i += 1
        current_reference = np.nonzero(f_distance_metric(X[current_id,:],X[set_id,:],metric_distance=distance_metric) < ref_sel_threshold)[1]
        
        if len(current_reference) == 0:
            set_id.append(current_id)

        del current_reference
        
    
    return X[set_id,:],time()-t0


'''
    Bauckhage C. Numpy/scipy Recipes for Data Science: k-Medoids Clustering[R]. Technical Report, University of Bonn, 2015.
'''
def kMedoids(X, parameters = {}):
    
    if("function_distance" in parameters):
        f_distance_metric = parameters["function_distance"]
    else:
        f_distance_metric = pairwise_cosine_distance
    if("distance_metric" in parameters):
        distance_metric = parameters["distance_metric"]
    else:
        distance_metric = "jaccard"
    
    if("k" in parameters):
        k = parameters["k"]
    else:
        k = 10
    
    if("tmax" in parameters):
        tmax = parameters["tmax"]
    else:
        tmax = 100
    
    t0 = time()
    if(distance_metric == "jaccard"):
        D = f_distance_metric(X.todense(),metric_distance=distance_metric)
    else:
        D = f_distance_metric(X,metric_distance=distance_metric)
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
    return X[M],time()-t0



def kmeans(X, parameters = {}):
    
    t0 = time()
    if("function_distance" in parameters):
        f_distance_metric = parameters["function_distance"]
    else:
        f_distance_metric = pairwise_cosine_distance
    if("distance_metric" in parameters):
        distance_metric = parameters["distance_metric"]
    else:
        distance_metric = "jaccard"
        
    if("k" in parameters):
        k = parameters["k"]
    else:
        k = 10
        
    if("random_state" in parameters):
        random_state = parameters["random_state"]
    else:
        random_state = 0
    
        
    #D = f_distance_metric(X).T
        
    
    kmeans = KMeans(n_clusters=k, random_state=random_state).fit(X)
    
    M = kmeans.cluster_centers_
    
    return np.array(M), time()-t0

def birch(X, parameters = {}):
    
    t0 = time()
    if("function_distance" in parameters):
        f_distance_metric = parameters["function_distance"]
    else:
        f_distance_metric = pairwise_cosine_distance
    if("distance_metric" in parameters):
        distance_metric = parameters["distance_metric"]
    else:
        distance_metric = "jaccard"
        
    if("k" in parameters):
        k = parameters["k"]
    else:
        k = 10
        
    if("random_state" in parameters):
        random_state = parameters["random_state"]
    else:
        random_state = 0
    
        
    #D = f_distance_metric(X).T
        
    
    birch = Birch(threshold=0.5, branching_factor=10, n_clusters=k, compute_labels=True, copy=True).fit(X)
    
    M = birch.subcluster_centers_
    
    return np.array(M), time()-t0
def kmedoidwv(X, parameters = {}):
    t0 = time()
    if("function_distance" in parameters):
        f_distance_metric = parameters["function_distance"]
    else:
        f_distance_metric = pairwise_cosine_distance
    if("random_state" in parameters):
        random_state = parameters["random_state"]
    else:
        random_state = 0
        
    if("k" in parameters):
        k = parameters["k"]
    else:
        k = 10
    
    dictParametersTokenized = parameters["pbinns__vocabulary"]
    word_vectors = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
#     word_vectors.get_vector("king")
#     model.most_similar(positive=[your_word_vector], topn=1)
    t0 = time()
    
    keys = list(dictParametersTokenized.keys())
    
    D = np.array([ word_vectors.get_vector(keys[x]) for x in range(keys.__len__()) ])
              
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

    indexX = [dictParametersTokenized(keys[m]) for m in M]
    # return results
    return X[indexX,:],time()-t0

