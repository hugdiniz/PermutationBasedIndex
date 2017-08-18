def pairwise_cosine_distance(Y,X):
    return pairwise_distances(Y,X, metric='cosine', n_jobs=1)