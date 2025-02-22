from PermutationBasedIndex.distance_metric import pairwise_cosine_distance
from math import ceil, sqrt, floor
import numpy as np
import scipy as sp
import random
from time import time
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import KMeans,Birch,MiniBatchKMeans,AffinityPropagation
from random import sample
from sklearn.decomposition import PCA
from pyclustering.cluster.kmedoids import kmedoids

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

def reference_set_selection_ordered(X,parameters = {}):
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
    
    set_id = np.array([ceil(X.shape[0]/2)])

    current_id = set_id[0]
    
    first_reference = np.nonzero(f_distance_metric(X[current_id,:],X,metric_distance=distance_metric) > ref_sel_threshold)[1]

    i = 0 
    set_distances = [0]       
    while i < len(first_reference):
        current_id = first_reference[i]
        i += 1
        distances = f_distance_metric(X[current_id,:],X[set_id,:],metric_distance=distance_metric)
        current_reference = np.nonzero(distances < ref_sel_threshold)[1]
        
        if len(current_reference) == 0:
            set_distances += distances[0]            
            set_distances = np.append(set_distances,sum(distances[0]))            
            set_id = np.append(set_id,current_id)
        del current_reference
        
    
    return X[set_id[np.argsort(-distances[0])[:reference_set_size]],:],time()-t0


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
        
    if("max_iter" in parameters):
        max_iter = parameters["max_iter"]
    else:
        max_iter = 100
            
    if("k" in parameters):
        k = parameters["k"]
    else:
        k = 10
        
    if("random_state" in parameters):
        random_state = parameters["random_state"]
    else:
        random_state = 0
    
        
    #D = f_distance_metric(X).T
        
    
    kmeans = AffinityPropagation(n_clusters=k, random_state=random_state,max_iter=max_iter).fit(X)
    
    M = kmeans.cluster_centers_
    
    return  kmeans.cluster_centers_, time()-t0


def aff_prop(X, parameters = {}):

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
        
    if("max_iter" in parameters):
        max_iter = parameters["max_iter"]
    else:
        max_iter = 200
    
        
    #D = f_distance_metric(X).T
        
    
    kmeans = AffinityPropagation(max_iter=max_iter).fit(X)
   
    
    return kmeans.cluster_centers_, time()-t0


def affinityPropagation(X, parameters = {}):
    
    t0 = time()
    if("function_distance" in parameters):
        f_distance_metric = parameters["function_distance"]
    else:
        f_distance_metric = pairwise_cosine_distance
    if("distance_metric" in parameters):
        distance_metric = parameters["distance_metric"]
    else:
        distance_metric = "jaccard"
        
    if("max_iter" in parameters):
        max_iter = parameters["max_iter"]
    else:
        max_iter = 100
            
    if("k" in parameters):
        k = parameters["k"]
    else:
        k = 10
        
    if("random_state" in parameters):
        random_state = parameters["random_state"]
    else:
        random_state = 0
    
        
    #D = f_distance_metric(X).T
        
    
    kmeans = AffinityPropagation().fit(X)
    
    M = kmeans.cluster_centers_
    
    return  kmeans.cluster_centers_, time()-t0

def miniBatchkmeans(X, parameters = {}):
    
    t0 = time()
    if("function_distance" in parameters):
        f_distance_metric = parameters["function_distance"]
    else:
        f_distance_metric = pairwise_cosine_distance
    if("distance_metric" in parameters):
        distance_metric = parameters["distance_metric"]
    else:
        distance_metric = "jaccard"
        
    if("max_iter" in parameters):
        max_iter = parameters["max_iter"]
    else:
        max_iter = 100
            
    if("k" in parameters):
        k = parameters["k"]
    else:
        k = 10
        
    if("random_state" in parameters):
        random_state = parameters["random_state"]
    else:
        random_state = 0
    
        
    #D = f_distance_metric(X).T
        
    
    kmeans = MiniBatchKMeans(n_clusters=k).fit(X)
    
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


def kmedoidwv_old(X, parameters = {}):
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
        
    if("tmax" in parameters):
        tmax = parameters["tmax"]
    else:
        tmax = 100
        
    if("distance_metric" in parameters):
        distance_metric = parameters["distance_metric"]
    else:
        distance_metric = "jaccard"
    
    t0 = time()
    
    keys = parameters["pbinns__words"]
    t1 = time()
#     if(distance_metric == "jaccard"):
#         
#         D = f_distance_metric(pca.fit_transform(parameters["pbinns__words_in_vec"]),metric_distance=distance_metric)
#     else:
#         D = f_distance_metric(parameters["pbinns__words_in_vec"].todense())
    D = parameters["pbinns__words_in_vec"]
    
    m, n = D.shape
    print("time to create a distance_metric"+str(time() - t1))
    
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


import sys
def kmedoidwv(X, parameters = {}):    
    
    sys.setrecursionlimit(1500000)
    if("k" in parameters):
        k = parameters["k"]
    else:
        k = 10
        
    if("tmax" in parameters):
        tmax = parameters["tmax"]
    else:
        tmax = 100
        
    
    matrixww = parameters["pbinns__words_in_vec"]
    words = parameters["pbinns__words"]
    vocabulary_ = parameters["pbinns__vocabulary"]
    
    t0 = time()
    randArray = np.array(words[:100])
    np.random.shuffle(randArray)
    randArray = randArray[:k]
    
    kmModel = kmedoids(matrixww.toarray(),randArray)
    kmModel.process()
    
    medoids = kmModel.get_medoids()
    
    original_index = [vocabulary_[words[medoid]] for medoid in medoids]
    
    return X[original_index,:],time()-t0
    
    
def psis(X,parameters = {}):
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
        ref_sel_threshold = 1200
    
    t0 = time()
    ids = []
    
    matrixDistance = np.array(f_distance_metric(X))
    
    size_id = np.zeros(reference_set_size)    
    current_id = ceil(matrixDistance.shape[0]/2)
    except_current_id = np.arange(matrixDistance.shape[0])!=current_id
    ids.append(np.nonzero(matrixDistance[current_id,except_current_id] >= ref_sel_threshold)[0][0])

    i = 1        
    while i < matrixDistance[:,:].shape[0]  and ids.__len__() < reference_set_size:
        if(np.all(matrixDistance[ids,i] >= ref_sel_threshold)):
            ids.append(i)
        i = i + 1
    
    return X[ids,:],time()-t0
    
    

def reference_set_selection_wv(X,parameters = {}):
    '''
        Distributed Selection: close reference points are neglected based on a threshold
         
    ''' 

    '''
        randomly selects the first reference point
    '''
    
    matrixww = parameters["pbinns__words_in_vec"]
    words = parameters["pbinns__words"]
    vocabulary_ = parameters["pbinns__vocabulary"]
    
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
        ref_sel_threshold = 0.04
    
    t0 = time()
    ids = []
    
    size_id = np.zeros(reference_set_size)    
    current_id = ceil(matrixww.shape[0]/2)
    except_current_id = np.arange(matrixww.shape[0])!=current_id
    ids.append(np.nonzero(matrixww[current_id,except_current_id] >= ref_sel_threshold)[1][0])

    i = 1        
    while i < matrixww[:,:].shape[0]  and ids.__len__() < reference_set_size:
        if(np.all(matrixww[ids,i].toarray() >= ref_sel_threshold)):
            ids.append(i)
        i = i + 1
      
        
    
    original_index = [] #[vocabulary_[words[medoid]] for medoid in set_id]
    
    for medoid in ids:
        print(words[medoid])
        if words[medoid] in vocabulary_.keys():
            original_index.append(vocabulary_[words[medoid]])
    
    return X[original_index,:],time()-t0

    
    
def f_similarity_matrix(w,model):
    a = np.zeros([w.size,w.size])    
    for i in range(w.size):
        a[i,i:] = np.where(True,model.similarity(w[i:],w[i]),0)
    return a





