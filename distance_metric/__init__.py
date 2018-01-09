from sklearn.metrics.pairwise import pairwise_distances
import numpy as np

def pairwise_cosine_distance(Y,X = [], metric_distance = 'cosine'):
    if(X != []):
        return pairwise_distances(Y,X, metric=metric_distance, n_jobs=1)
    else:
        return pairwise_distances(Y, metric=metric_distance, n_jobs=1)
 

