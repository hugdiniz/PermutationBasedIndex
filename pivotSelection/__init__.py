from PermutationBasedIndex.distance_metric import pairwise_cosine_distance

def reference_set_selection(X,ref_sel_threshold = 0.5, metric_vector_distance = pairwise_cosine_distance):
    '''
        Distributed Selection: close reference points are neglected based on a threshold
         
    ''' 

    '''
        randomly selects the first reference point
    '''
#         set_id = sample(range(X.shape[0]),1)
    
    set_id = [ceil(X.shape[0]/2)]

    current_id = set_id[0]
    
    first_reference = np.nonzero(metric_vector_distance(X[current_id,:],X) > ref_sel_threshold)[1]

    i = 0        
    while len(set_id) < self.reference_set_size and i < len(first_reference):
        current_id = first_reference[i]
        i += 1
        current_reference = np.nonzero(metric_vector_distance(X[current_id,:],X[set_id,:]) < self.ref_sel_threshold)[1]
        
        if len(current_reference) == 0:
            set_id.append(current_id)

        del current_reference
    return set_id