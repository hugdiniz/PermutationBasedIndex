from PermutationBasedIndex.distance_metric import pairwise_cosine_distance
from PermutationBasedIndex.pivotSelection import reference_set_selection

def relative_ordered_list(X, d_index, index_features, reference_set_id, prunning_size, metric_vector_distance = pairwise_cosine_distance):
        
    t0 = time()
    
#         print(X.shape)
    distances = metric_vector_distance(X[d_index,:],index_features[reference_set_id,:])
    
    lr = argsort(distances,axis=1)
    return lr[:,:prunning_size], time() - t0
    
def index_collection(X,bucket_count, prunning_size, metric_vector_distance = pairwise_cosine_distance):
    """ inverted index (ii) creation 

    Parameters
    ----------
    X : sparse matrix, [n_samples, n_features]
    """
    collection_size = X.shape[0]
    ii = collections.defaultdict(lambda : collections.defaultdict(list))
    
    reference_set_id = reference_set_selection(X)
    index_features = X
     
    bij = np.empty((X.shape[0],len(reference_set_id)),np.int)
    for d_index in range(X.shape[0]):
        d_list_in_r, d_list_time = relative_ordered_list(X, d_index, index_features, reference_set_id, prunning_size, metric_vector_distance)

        t0 = time()
        for j in range(d_list_in_r.shape[1]):
            bij[d_index,j] = ceil(((bucket_count-1)*d_list_in_r[0,j])/len(reference_set_id))
            ii[j][bij[d_index,j]].append(d_index) 
        
        
        
    return bij,ii
        
def score_queries(X, bij, ii ,collection_size,prunning_size,bucket_count,metric_vector_distance = pairwise_cosine_distance):
    """ scores(count) collection document collisions against queries features  

    Parameters
    ----------
    X : sparse matrix, [n_samples, n_features]
    """

    index_features = X
    reference_set_id = reference_set_selection(X)
    collection_size = X.shape[0]
    
    scores = np.zeros((X.shape[0], collection_size))
    
    time_to_score = 0
    
    for q_index in range(X.shape[0]):
        q_list_in_r, q_list_time = relative_ordered_list(X, q_index, index_features, reference_set_id, prunning_size, metric_vector_distance)

        t0 = time()
        for j in range(q_list_in_r.shape[1]):
            bqj = ceil(((bucket_count-1)*q_list_in_r[0,j])/len(reference_set_id))

            scores[q_index,ii[j][bqj-1]] = scores[q_index,ii[j][bqj-1]] + 1
            scores[q_index,ii[j][bqj]]   = scores[q_index,ii[j][bqj]] + 2
            scores[q_index,ii[j][bqj+1]] = scores[q_index,ii[j][bqj+1]] + 1

    return scores



