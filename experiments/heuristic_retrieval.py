import numpy as np
from time import time
import pandas as pd

from datasets.extractors import short_plagiarised_answers_extractor, pan_plagiarism_corpus_2010_extractor, pan_plagiarism_corpus_2011_extractor
from locality_sensitive_hashing import LSHTransformer, minmax_hashing, LSHIINearestNeighbors,\
    InvertedIndexNearestNeighbors,\
    MinMaxSymetricFPRAE, MinMaxSymetricFPRAP,\
    MinMaxSymetricDistributedFP, MinMaxAsymetricFPRAE, MinMaxAsymetricFPRAP,\
    MinMaxAsymetricDistributedFP, min_hashing, minmaxCSALowerBound_hashing,\
    minmaxCSAFullBound_hashing, justCSALowerBound_hashing,\
    justCSAFullBound_hashing
    
from PermutationBasedIndex import PBINearestNeighbors
from PermutationBasedIndex.pivotSelection import reference_set_selection, kMedoids, kmeans,random_select_pivot
from PermutationBasedIndex.experiments import *




if __name__ == '__main__':
  if __name__ == '__main__':
    
    '''
        creating TfIdfVectorizer, LSHTransformer and LSHIINearestNeighbors parameters grids and
        storing it as pandas Dataframes on hdf
    '''
#    dataset_name = "psa"
#     dataset_name = "pan10"
    dataset_name = "pan11"

    #dataset_name,sample_size = "pan10-%d-samples",10 
    #dataset_name = dataset_name%(sample_size)
    queries_percentage = 100
     
    cv_parameters = {
        "cv__analyzer" : ('word',),
        "cv__ngram_range" : (
                            (1,1),#                            
                             ),
        "cv__tokenizer" : (
                            None,#                           
                           ),
        "cv__lowercase" : (True,),
        "cv__min_df" : (
                         1,
                        ),
        "cv__binary" : (False,),
        "cv__stop_words" : ('english',),
        "cv__use_idf" : (True,),
        "cv__norm" : (
                      'l1',
                      ),
        
    }
    
    lsht_parameters = {
        "lsht__n_permutations" : (6, 192),
        "lsht__selection_function" : (
                                     MinMaxSymetricFPRAE(n_partitions=4),                                  
                                      ),
        "lsht__n_jobs" : (
                        -1,
                          )                       
    }
    
    lshnns_parameters = {
        "lshnns__n_neighbors" : (7983,),
        "lshnns__sort_neighbors" : (False,),
                         
    }
    
    nns_parameters = {
        "nns__n_neighbors" : lshnns_parameters['lshnns__n_neighbors'],
        "nns__sort_neighbors" : lshnns_parameters['lshnns__sort_neighbors'],
    }

    pbinns_parameters = {
        "pbinns__n_neighbors" : nns_parameters['nns__n_neighbors'],
        "pbinns__sort_neighbors" : nns_parameters['nns__sort_neighbors'],
        "pbinns__bucket_count" : (25,),
        "pbinns__prunning_size" : (100,),
        "pbinns__pivot_parameters" : (
        
        json.dumps({          
            "pivot_selection_function" :kMedoids.__name__ ,
            "k" : 25,
            "tmax":100,            
        }),
#         json.dumps({            
#              "pivot_selection_function" :kmeans.__name__ ,
#              "k" : 25,            
#          }),
#         json.dumps({          
#             "pivot_selection_function" :random_select_pivot.__name__ ,
#             "k" : 25,
#                         
#         }),
#         json.dumps({            
#             "pivot_selection_function" :reference_set_selection.__name__ ,
#             "k" : 25,
#             "ref_sel_threshold" : 0.5,            
#         }),
    ),
        
    }


    '''
        storing parameters dataframes 
    '''

    parameters_sequence = [('cv',cv_parameters),('lsht',lsht_parameters),('lshnns',lshnns_parameters)]
    parameters_grids_list = generate_or_load_parameters_grids(parameters_sequence,dataset_name)

    cv_df_paramaters, lsht_df_paramaters, lshnns_df_paramaters = parameters_grids_list

#     for i in parameters_grids_list:
#         print(i)             
#         print('xxxxxxxxxxxxxxxxxxxx')
#     exit()
    
    '''
        nearest neighbor search without LSH
    '''    
    parameters_sequence = [('cv',cv_parameters),('nns',nns_parameters)]
    parameters_grids_list = generate_or_load_parameters_grids(parameters_sequence,dataset_name)

    nns_df_paramaters = parameters_grids_list[1]

#    for i in parameters_grids_list:
#        print(i)             
#        print('xxxxxxxxxxxxxxxxxxxx')
#    exit() 

    '''
        permutation-Based Index(PBI) nearest neighbor search
    '''    

    parameters_sequence = [('cv',cv_parameters),('pbinns',pbinns_parameters)]
    parameters_grids_list = generate_or_load_parameters_grids(parameters_sequence,dataset_name)

    pbinns_df_paramaters = parameters_grids_list[1]
    
#     for i in parameters_grids_list:
#         print(i)             
#         print('xxxxxxxxxxxxxxxxxxxx')
#     exit()
    

    '''
        dataset extraction
    '''
    if dataset_name == "psa":
        corpus_name, (suspicious_info, source_info,target, dataset_encoding) = dataset_name, short_plagiarised_answers_extractor.load_as_ir_task()
    elif dataset_name == "pan10":
        corpus_name, (suspicious_info, source_info,target, dataset_encoding) = dataset_name, pan_plagiarism_corpus_2010_extractor.load_as_ir_task(allow_queries_without_relevants=False)
    elif dataset_name == "pan11":
        corpus_name, (suspicious_info, source_info,target, dataset_encoding) = dataset_name, pan_plagiarism_corpus_2011_extractor.load_as_ir_task(allow_queries_without_relevants=False, language_filter="EN")
    elif "pan10" in dataset_name and "-samples" in dataset_name:
        corpus_name, (suspicious_info, source_info,target, dataset_encoding) = dataset_name, pan_plagiarism_corpus_2010_extractor.load_sample_as_ir_task(sample_size, language_filter="EN")

    print('queries:',suspicious_info.shape,' Documents:',source_info.shape)
    queries_number = int((suspicious_info.shape[0] / 100) * queries_percentage)
    
    
    documents = Parallel(n_jobs=-1,backend="threading",verbose=1)(delayed(encode_dataframe_content)(si, dataset_encoding) for si in source_info['content'].values)
    queries = Parallel(n_jobs=-1,backend="threading",verbose=1)(delayed(encode_dataframe_content)(si, dataset_encoding) for si in suspicious_info['content'].values)[:queries_number]
    target = target[:queries_number,:]
    del suspicious_info, source_info
    
    print(nns_df_paramaters)
    print(lsht_df_paramaters) 
        
#     exit()
    
    '''
        using scikit-learn : tokenization
    '''

    for i,linei in cv_df_paramaters.iterrows():
        tokenize_by_parameters(documents,queries,target,dataset_name,linei,i,dataset_encoding,dataset_encoding)

    queries_count,documents_count = target.shape
    del documents, queries, target
      
    '''
        nearest neighbor search (ranking)
    '''
    
    for i,linei in pbinns_df_paramaters.iterrows():
        print("#"*10+" PBI N.N.S. "+"#"*10)
        print(linei)
        pbinearest_neighbors_search(dataset_name,linei,i,dataset_encoding)
        print("-"*20)
    
    today = datetime.now()
    today = today.strftime('%Y-%m-%d_%H-%M-%S_')
  
#     '''
#         Permutation-Based Index (PBI) logging nearest neighbors results on csv
#     '''
    print_pbi(cv_df_paramaters, pbinns_df_paramaters,dataset_name,documents_count,queries_count) 
