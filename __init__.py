from PermutationBasedIndex.distance_metric import pairwise_cosine_distance
from PermutationBasedIndex.pivotSelection import reference_set_selection, kMedoids, kmeans
import collections
from math import ceil, sqrt, floor
import numpy as np
from scipy import argsort
from locality_sensitive_hashing import InvertedIndex,InvertedIndexNearestNeighborsBaseEstimator
from time import time
from sklearn.metrics.pairwise import pairwise_distances

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



'''
    Permutation Based Index !
'''
class PermutationBasedIndex(InvertedIndex):
    def __init__(self, bucket_count, reference_set_size, prunning_size, ref_sel_threshold = 0.5):
        self.bucket_count,self.reference_set_size,self.prunning_size,self.ref_sel_threshold = bucket_count,reference_set_size,prunning_size,ref_sel_threshold

    def reference_set_selection(self,X):
        '''
            Distributed Selection: close reference points are neglected based on a threshold
             
        '''
        t0 = time()
        
        '''
            randomly selects the first reference point
        '''
#         set_id = sample(range(X.shape[0]),1)
        
        set_id = [ceil(X.shape[0]/2)]

        current_id = set_id[0]
        
        first_reference = np.nonzero(pairwise_distances(X[current_id,:],X, metric='cosine', n_jobs=1) > self.ref_sel_threshold)[1]

        i = 0        
        while len(set_id) < self.reference_set_size and i < len(first_reference):
            current_id = first_reference[i]
            i += 1
            current_reference = np.nonzero(pairwise_distances(X[current_id,:],X[set_id,:], metric='cosine', n_jobs=1) < self.ref_sel_threshold)[1]
            
            if len(current_reference) == 0:
                set_id.append(current_id)

            del current_reference
        
#         print(set_id)
#         print(len(set_id),' x ', self.reference_set_size)
#         set_id = sample(range(X.shape[0]),self.reference_set_size)
        return X[set_id,:], time()-t0
    
    def relative_ordered_list(self, X, d_index):
        
        t0 = time()
        
#         print(X.shape)
        distances = pairwise_distances(X[d_index,:],self.reference_set_id, metric='cosine', n_jobs=1)
        
        lr = argsort(distances,axis=1)
        return lr[:,:self.prunning_size], time() - t0
    
    def index_collection(self,X):
        """ inverted index (ii) creation 

        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
        """
        self.collection_size = X.shape[0]
        self.ii = collections.defaultdict(lambda : collections.defaultdict(list))
        
        self.reference_set_id, self.index_time = kmeans(X)
        self.index_features = X
         
        self.bij = np.empty((X.shape[0],self.reference_set_id.shape[0]),np.int)
        for d_index in range(X.shape[0]):
            d_list_in_r, d_list_time = self.relative_ordered_list(X, d_index)

            t0 = time()
            for j in range(d_list_in_r.shape[1]):
                self.bij[d_index,j] = ceil(((self.bucket_count-1)*d_list_in_r[0,j])/self.reference_set_id.shape[0])
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
                bqj = ceil(((self.bucket_count-1)*q_list_in_r[0,j])/self.reference_set_id.shape[0])

                scores[q_index,self.ii[j][bqj-1]] = scores[q_index,self.ii[j][bqj-1]] + 1
                scores[q_index,self.ii[j][bqj]] = scores[q_index,self.ii[j][bqj]] + 2
                scores[q_index,self.ii[j][bqj+1]] = scores[q_index,self.ii[j][bqj+1]] + 1
                    
            
            time_to_score += time() - t0
            time_to_score += q_list_time 

        return scores, time_to_score
        
class PBINearestNeighbors(InvertedIndexNearestNeighborsBaseEstimator):
    def __init__(self, bucket_count, reference_set_size, prunning_size, ref_sel_threshold, n_neighbors, sort_neighbors):
        self.bucket_count, self.reference_set_size, self.prunning_size, self.ref_sel_threshold = bucket_count, reference_set_size, prunning_size, ref_sel_threshold
        InvertedIndexNearestNeighborsBaseEstimator.__init__(self, self.create_inverted_index(), n_neighbors, sort_neighbors)

    def create_inverted_index(self):
        return PermutationBasedIndex(self.bucket_count, self.reference_set_size, self.prunning_size, self.ref_sel_threshold)    
        
    def set_params(self, **params):
        InvertedIndexNearestNeighborsBaseEstimator.set_params(self,**params)
        self.ii = self.create_inverted_index()