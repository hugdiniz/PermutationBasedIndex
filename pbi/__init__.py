from PermutationBasedIndex.distance_metric import pairwise_cosine_distance
from PermutationBasedIndex.pivotSelection import reference_set_selection
import collections
from math import ceil, sqrt, floor
import numpy as np
from scipy import argsort

def relative_ordered_list(X, d_index, index_features, reference_set_id, prunning_size, metric_vector_distance = pairwise_cosine_distance):
            
#         print(X.shape)
    distances = metric_vector_distance(X[d_index,:],index_features[reference_set_id,:])
    
    lr = argsort(distances,axis=1)
    return lr[:,:prunning_size]
    
def index_collection(X,bucket_count, prunning_size, metric_vector_distance = pairwise_cosine_distance, pivot_selection_function = reference_set_selection):
    """ inverted index (ii) creation 

    Parameters
    ----------
    X : sparse matrix, [n_samples, n_features]
    """
    collection_size = X.shape[0]
    ii = collections.defaultdict(lambda : collections.defaultdict(list))
    
    reference_set_id = pivot_selection_function(X)
    index_features = X
     
    bij = np.empty((X.shape[0],len(reference_set_id)),np.int)
    for d_index in range(X.shape[0]):
        d_list_in_r = relative_ordered_list(X, d_index, index_features, reference_set_id, prunning_size, metric_vector_distance)

        
        for j in range(d_list_in_r.shape[1]):
            bij[d_index,j] = ceil(((bucket_count-1)*d_list_in_r[0,j])/len(reference_set_id))
            ii[j][bij[d_index,j]].append(d_index) 
        
        
        
    return bij,ii
        
def score_queries(X, bij, ii ,prunning_size,bucket_count,metric_vector_distance = pairwise_cosine_distance):
    """ scores(count) collection document collisions against queries features  

    Parameters
    ----------
    X : sparse matrix, [n_samples, n_features]
    """

    index_features = X
    reference_set_id = reference_set_selection(X)
    collection_size = X.shape[0]
    
    scores = np.zeros((X.shape[0], collection_size))
    
    
    
    for q_index in range(X.shape[0]):
        q_list_in_r = relative_ordered_list(X, q_index, index_features, reference_set_id, prunning_size, metric_vector_distance)

        
        for j in range(q_list_in_r.shape[1]):
            bqj = ceil(((bucket_count-1)*q_list_in_r[0,j])/len(reference_set_id))

            scores[q_index,ii[j][bqj-1]] = scores[q_index,ii[j][bqj-1]] + 1
            scores[q_index,ii[j][bqj]]   = scores[q_index,ii[j][bqj]] + 2
            scores[q_index,ii[j][bqj+1]] = scores[q_index,ii[j][bqj+1]] + 1

    return scores



