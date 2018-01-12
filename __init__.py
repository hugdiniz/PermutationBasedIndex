from PermutationBasedIndex.distance_metric import pairwise_cosine_distance
from PermutationBasedIndex.pivotSelection import reference_set_selection, kMedoids, kmeans
import collections
from math import ceil, sqrt, floor
import numpy as np
from scipy import argsort
from locality_sensitive_hashing import InvertedIndex,InvertedIndexNearestNeighborsBaseEstimator
from time import time
from sklearn.metrics.pairwise import pairwise_distances


class PermutationBasedIndex(InvertedIndex):
    def __init__(self, bucket_count, reference_set_size, prunning_size, ref_sel_threshold = 0.5):
        self.bucket_count,self.reference_set_size,self.prunning_size,self.ref_sel_threshold = bucket_count,reference_set_size,prunning_size,ref_sel_threshold
  
    
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
        
        self.reference_set_id, self.index_time = kMedoids(X)
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