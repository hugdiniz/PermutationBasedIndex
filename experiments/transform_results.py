from PermutationBasedIndex.experiments import *
import json
from math import ceil
import numpy as np


from datasets.extractors import short_plagiarised_answers_extractor, pan_plagiarism_corpus_2010_extractor, pan_plagiarism_corpus_2011_extractor
from locality_sensitive_hashing import LSHTransformer, minmax_hashing, LSHIINearestNeighbors,\
    InvertedIndexNearestNeighbors,\
    MinMaxSymetricFPRAE, MinMaxSymetricFPRAP,\
    MinMaxSymetricDistributedFP, MinMaxAsymetricFPRAE, MinMaxAsymetricFPRAP,\
    MinMaxAsymetricDistributedFP, min_hashing, minmaxCSALowerBound_hashing,\
    minmaxCSAFullBound_hashing, justCSALowerBound_hashing,\
    justCSAFullBound_hashing

from PermutationBasedIndex.experiments import *
from PermutationBasedIndex.pivotSelection import reference_set_selection, kMedoids, kmeans,random_select_pivot


if __name__ == '__main__':
    
     
    '''
        creating TfIdfVectorizer, LSHTransformer and LSHIINearestNeighbors parameters grids and
        storing it as pandas Dataframes on hdf
    '''
    #    dataset_name = "psa"
    #     dataset_name = "pan10"
    #    dataset_name = "pan11"
    
    dataset_name,sample_size = "pan10-%d-samples",10 
    dataset_name = dataset_name%(sample_size)
    queries_percentage = 100    
    
    print_technique(0,0,dataset_name)
    #print_pbi(cv_df_paramaters, pbinns_df_paramaters,dataset_name,documents_count,queries_count)
    