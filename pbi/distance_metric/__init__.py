from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from sklearn.utils import shuffle

def pairwise_cosine_distance(Y,X, metric_distance = 'cosine'):
    return pairwise_distances(Y,X, metric=metric_distance, n_jobs=1)

def minmax_hashing(element_set, permutation):
    rows,_ = np.nonzero(element_set)
    min_id,max_id = np.Inf,-np.Inf
    
    for linei in rows:
        perm_id = permutation[linei]
        if perm_id < min_id:
            min_id = perm_id
        if perm_id > max_id:
            max_id = perm_id

    return np.array([min_id,max_id])

def lsh_cosine_distance(Y,X,lsh = minmax_hashing, pairwise_distances_metric = pairwise_cosine_distance, permutation_count = 100):
    indexes_permutations = [shuffle([i+1 for i in range(all_fingerprints.shape[0])]) for j in range(permutation_count)]    
    for j in range(all_fingerprints.shape[1]):
       
        current_document = all_fingerprints[:,j]
        
        for i in range(approach_permutation_count):
            pi_results = lsh(current_document, indexes_permutations[i])
            approach_finger[j,selection_size*i:selection_size*(i+1)] = pi_results
 
        del current_document
    
    approach_jaccard_sim = pairwise_distances_metric(approach_finger)
        
    return approach_jaccard_sim



