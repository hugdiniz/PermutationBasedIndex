from PermutationBasedIndex.distance_metric import pairwise_cosine_distance

def relative_ordered_list(X, d_index, index_features, reference_set_id, prunning_size, metric_vector_distance = pairwise_cosine_distance):
        
    t0 = time()
    
#         print(X.shape)
    distances = metric_vector_distance(X[d_index,:],index_features[reference_set_id,:])
    
    lr = argsort(distances,axis=1)
    return lr[:,:prunning_size], time() - t0
    
def index_collection(self,X):
    """ inverted index (ii) creation 

    Parameters
    ----------
    X : sparse matrix, [n_samples, n_features]
    """
    self.collection_size = X.shape[0]
    self.ii = collections.defaultdict(lambda : collections.defaultdict(list))
    
    self.reference_set_id, self.index_time = self.reference_set_selection(X)
    self.index_features = X
     
    self.bij = np.empty((X.shape[0],len(self.reference_set_id)),np.int)
    for d_index in range(X.shape[0]):
        d_list_in_r, d_list_time = self.relative_ordered_list(X, d_index)

        t0 = time()
        for j in range(d_list_in_r.shape[1]):
            self.bij[d_index,j] = ceil(((self.bucket_count-1)*d_list_in_r[0,j])/len(self.reference_set_id))
            self.ii[j][self.bij[d_index,j]].append(d_index) 
        
        self.index_time += time() - t0
        self.index_time += d_list_time 
        
    return self.index_time
        
def score_queries(self,X):
    """ scores(count) collection document collisions against queries features  

    Parameters
    ----------
    X : sparse matrix, [n_samples, n_features]
    """


    scores = np.zeros((X.shape[0], self.collection_size))
    
    time_to_score = 0
    
    for q_index in range(X.shape[0]):
        q_list_in_r, q_list_time = self.relative_ordered_list(X, q_index)

        t0 = time()
        for j in range(q_list_in_r.shape[1]):
            bqj = ceil(((self.bucket_count-1)*q_list_in_r[0,j])/len(self.reference_set_id))

            scores[q_index,self.ii[j][bqj-1]] = scores[q_index,self.ii[j][bqj-1]] + 1
            scores[q_index,self.ii[j][bqj]]   = scores[q_index,self.ii[j][bqj]] + 2
            scores[q_index,self.ii[j][bqj+1]] = scores[q_index,self.ii[j][bqj+1]] + 1
                
        
        time_to_score += time() - t0
        time_to_score += q_list_time 

    return scores, time_to_score



