import numpy as np
from time import time
import pandas as pd
from math import ceil

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
    
    '''
        creating TfIdfVectorizer, LSHTransformer and LSHIINearestNeighbors parameters grids and
        storing it as pandas Dataframes on hdf
    '''
#    dataset_name = "psa"
#    dataset_name = "pan10"
    dataset_name = "pan11"


#     dataset_name,sample_size = "pan10-%d-samples",10 
#     dataset_name = dataset_name%(sample_size)
    queries_percentage = 25
     
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
        "lsht__n_permutations" : (6,),
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
        "pbinns__bucket_count" : (80,),
        "pbinns__prunning_size" : (ceil(nns_parameters['nns__n_neighbors'][0]*25/100),),
        "pbinns__using_lsh" : (False,),
        "pbinns__pivot_parameters" : (
            
        json.dumps({          
            "pivot_selection_function" :kMedoids.__name__ ,
            "k" : 20,
            "tmax":10000,            
        }),
                                      
        json.dumps({          
            "pivot_selection_function" :kMedoids.__name__ ,
            "k" : 30,
            "tmax":10000,            
        }),
#          
#         json.dumps({          
#             "pivot_selection_function" :kMedoids.__name__ ,
#             "k" : ceil(nns_parameters['nns__n_neighbors'][0]*25/100),
#             "tmax":500,            
#         }),
#          
#         json.dumps({          
#             "pivot_selection_function" :kMedoids.__name__ ,
#             "k" : ceil(nns_parameters['nns__n_neighbors'][0]*50/100),
#             "tmax":500,            
#         }),
#         
#         json.dumps({          
#             "pivot_selection_function" :kMedoids.__name__ ,
#             "k" : ceil(nns_parameters['nns__n_neighbors'][0]*75/100),
#             "tmax":500,            
#         }),
                                      
#         json.dumps({          
#             "pivot_selection_function" :kmeans.__name__ ,
#             "k" : ceil(nns_parameters['nns__n_neighbors'][0]*10/100),                     
#         }),
#          
       
#          
#         json.dumps({          
#             "pivot_selection_function" :kmeans.__name__ ,
#             "k" : ceil(nns_parameters['nns__n_neighbors'][0]*50/100),                   
#         }),
#         
#         json.dumps({          
#             "pivot_selection_function" :kmeans.__name__ ,
#             "k" : ceil(nns_parameters['nns__n_neighbors'][0]*75/100),                       
#         }),                             
#         
#         json.dumps({          
#             "pivot_selection_function" :random_select_pivot.__name__ ,
#             "k" : ceil(nns_parameters['nns__n_neighbors'][0]*25/100),                     
#         }),
#                                       
#         json.dumps({          
#             "pivot_selection_function" :random_select_pivot.__name__ ,
#             "k" : ceil(nns_parameters['nns__n_neighbors'][0]*25/1000),                     
#         }),
#          
#         json.dumps({          
#             "pivot_selection_function" :random_select_pivot.__name__ ,
#             "k" : ceil(nns_parameters['nns__n_neighbors'][0]*25/100),                       
#         }),
#          
        json.dumps({          
            "pivot_selection_function" :random_select_pivot.__name__ ,
            "k" : 50,                   
        }),
#         
#         json.dumps({          
#             "pivot_selection_function" :random_select_pivot.__name__ ,
#             "k" : ceil(nns_parameters['nns__n_neighbors'][0]*75/100),                       
#         }),      
# #                                       
#         json.dumps({            
#             "pivot_selection_function" :reference_set_selection.__name__ ,
#             "k" : ceil(nns_parameters['nns__n_neighbors'][0]*75/100),
#             "ref_sel_threshold" : 0.2,            
#         }),
#         json.dumps({            
#             "pivot_selection_function" :reference_set_selection.__name__ ,
#             "k" : ceil(nns_parameters['nns__n_neighbors'][0]*75/100),
#             "ref_sel_threshold" : 0.25,            
#         }),
# 
#         json.dumps({            
#             "pivot_selection_function" :reference_set_selection.__name__ ,
#             "k" : ceil(nns_parameters['nns__n_neighbors'][0]*25/100),
#             "ref_sel_threshold" : 0.5,            
#         }),
#         json.dumps({            
#             "pivot_selection_function" :reference_set_selection.__name__ ,
#             "k" : ceil(nns_parameters['nns__n_neighbors'][0]*75/100),
#             "ref_sel_threshold" : 0.5,            
#         }),
#         json.dumps({            
#             "pivot_selection_function" :reference_set_selection.__name__ ,
#             "k" : ceil(nns_parameters['nns__n_neighbors'][0]*25/100),
#             "ref_sel_threshold" : 0.75,            
#         }),
#         json.dumps({            
#             "pivot_selection_function" :reference_set_selection.__name__ ,
#             "k" : ceil(nns_parameters['nns__n_neighbors'][0]*75/100),
#             "ref_sel_threshold" : 0.75,            
#         }),
),
                         }


    '''
        storing parameters dataframes 
    '''
      
    parameters_sequence = [('cv',cv_parameters),('lsht',lsht_parameters),('lshnns',lshnns_parameters)]
    parameters_grids_list = generate_or_load_parameters_grids(parameters_sequence,dataset_name)

    cv_df_paramaters, lsht_df_paramaters, lshnns_df_paramaters = parameters_grids_list

    
    '''
        nearest neighbor search without LSH
    '''    
    parameters_sequence = [('cv',cv_parameters),('nns',nns_parameters)]
    parameters_grids_list = generate_or_load_parameters_grids(parameters_sequence,dataset_name)

    nns_df_paramaters = parameters_grids_list[1]

    '''
        permutation-Based Index(PBI) nearest neighbor search
    '''  
    parameters_sequence = [('cv',cv_parameters),('pbinns',pbinns_parameters)]
    parameters_grids_list = generate_or_load_parameters_grids(parameters_sequence,dataset_name)

    pbinns_df_paramaters = parameters_grids_list[1]    

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

#     for i,linei in lsht_df_paramaters.iterrows():
#         print(linei)
#         print('xxxxxx')
#         lsh_transform(dataset_name,linei,i,dataset_encoding)
# # 
# # 
#     for i,linei in lshnns_df_paramaters.iterrows():
#         print("#"*10+" LSH N.N.S. "+"#"*10)
#         print(linei)
#         lsh_nearest_neighbors_search(dataset_name,linei,i,dataset_encoding)
#         print("-"*20)

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
    
#     '''
#         logging LSH nearest neighbors results on csv
#     '''
    a = pd.merge(cv_df_paramaters, lsht_df_paramaters, how='inner', left_index=True, right_on=['input__filename_index',],)
    b = pd.merge(a, lshnns_df_paramaters, how='inner', left_index=True, right_on=['input__filename_index',],suffixes=('_lsht','_lshtnns'))    
    del a
 
    for rowi in b.iterrows():
        cv_index = rowi[1]['input__filename_index_lsht']
        lsht_index = rowi[1]['input__filename_index_lshtnns']
        nns_index = rowi[0]
         
#         print(cv_index,'-',lsht_index,'-',nns_index)
        cv_file_path = h5_results_filename(dataset_name, 'cv', cv_index).replace('results','time')
        lsht_file_path = h5_results_filename(dataset_name, 'lsht', lsht_index).replace('results','time')
        lshnns_file_path = h5_results_filename(dataset_name, 'lshnns', nns_index).replace('results','results_evaluation')
        lshnns_time_file_path = h5_results_filename(dataset_name, 'lshnns', nns_index).replace('results','time')
 
#         print('\t',cv_file_path)
#         print('\t',lsht_file_path)
#         print('\t',lshnns_file_path)
#         print('=========')
        approach_precisions = hdf_to_sparse_matrix('precisions', lshnns_file_path)
        approach_recalls = hdf_to_sparse_matrix('recalls', lshnns_file_path)
        average_precision = hdf_to_sparse_matrix('average_precisions', lshnns_file_path).todense()
         
        b.loc[nns_index,'MAP'] = average_precision.mean()
        b.loc[nns_index,'MAP_std'] = average_precision.std()
        b.loc[nns_index,'precision_recall_path'] = lshnns_file_path
        b.loc[nns_index,'recall_mean'] = approach_recalls[:,-1].todense().mean()
        b.loc[nns_index,'recall_std'] = approach_recalls[:,-1].todense().std()
        b.loc[nns_index,'precision_mean'] = approach_precisions[:,-1].todense().mean()
        b.loc[nns_index,'precision_std'] = approach_precisions[:,-1].todense().std()
         
        del approach_precisions, approach_recalls, average_precision
 
        b.loc[nns_index,'documents_count'] = documents_count
        b.loc[nns_index,'queries_count'] = queries_count
         
        with open(cv_file_path.replace('time.h5', 'vocabulary.pkl'),'rb') as f:
            b.loc[nns_index,'vocabulary_size'] = len(pickle.load(f))
 
        q = hdf_to_sparse_matrix('queries', lsht_file_path.replace('time','results'))
        b.loc[nns_index,'lsht_features'] = q.shape[1]
        del q
         
        b.loc[nns_index,'indexing_mean_time'] = 0
        b.loc[nns_index,'querying_mean_time'] = 0
         
        cv_time_dataframe = pd.read_hdf(cv_file_path, 'time_dataframe')
        b.loc[nns_index,'cv_documents_mean_time'] = cv_time_dataframe.loc[0,'documents_mean_time'] 
        b.loc[nns_index,'cv_queries_mean_time'] = cv_time_dataframe.loc[0,'queries_mean_time']
                 
        b.loc[nns_index,'indexing_mean_time'] += b.loc[nns_index,'cv_documents_mean_time']
        b.loc[nns_index,'querying_mean_time'] += b.loc[nns_index,'cv_queries_mean_time']
        del cv_time_dataframe 
 
        d_time = hdf_to_sparse_matrix('documents_time',lsht_file_path)
        q_time = hdf_to_sparse_matrix('queries_time',lsht_file_path)
         
        b.loc[nns_index,'lsht_documents_mean_time'] = d_time.sum(axis=1).mean() 
        b.loc[nns_index,'lsht_queries_mean_time'] = q_time.sum(axis=1).mean()
 
        b.loc[nns_index,'indexing_mean_time'] += b.loc[nns_index,'lsht_documents_mean_time']
        b.loc[nns_index,'querying_mean_time'] += b.loc[nns_index,'lsht_queries_mean_time']
        del d_time, q_time 
 
        nns_time_dataframe = pd.read_hdf(lshnns_time_file_path, 'time_dataframe')
        b.loc[nns_index,'nns_documents_mean_time'] = nns_time_dataframe.loc[0,'documents_mean_time'] 
        b.loc[nns_index,'nns_queries_mean_time'] = nns_time_dataframe.loc[0,'queries_mean_time']
 
        b.loc[nns_index,'indexing_mean_time'] += b.loc[nns_index,'nns_documents_mean_time']
        b.loc[nns_index,'querying_mean_time'] += b.loc[nns_index,'nns_queries_mean_time']
        del nns_time_dataframe
 
        print(b.loc[nns_index,'lsht__selection_function'],' : ',int(b.loc[nns_index,'lsht_features']),' features x ',b.loc[nns_index,'lsht__n_permutations'],' permutation')
        print("MAP = %4.2f[+-%4.2f]"%(b.loc[nns_index,'MAP'],b.loc[nns_index,'MAP_std']))
        print("recall = %4.2f[+-%4.2f]"%(b.loc[nns_index,'recall_mean'],b.loc[nns_index,'recall_std']))
        print("index time = %4.4f"%(b.loc[nns_index,'cv_documents_mean_time']+b.loc[nns_index,'lsht_documents_mean_time']+b.loc[nns_index,'nns_documents_mean_time']))
        print("query time = %4.4f"%(b.loc[nns_index,'cv_queries_mean_time']+b.loc[nns_index,'lsht_queries_mean_time']+b.loc[nns_index,'nns_queries_mean_time']))
          
        print("---->",b.loc[nns_index,'indexing_mean_time'])
    b.to_csv('%s%s_lsh_results.csv'%(today,dataset_name),sep='\t')
     
    del b
#     

