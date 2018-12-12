import numpy as np
import pickle
import os.path
from time import time
import pandas as pd
import json
import gensim
import re

from sklearn.pipeline import Pipeline
from sklearn.grid_search import ParameterGrid
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.externals.joblib.parallel import delayed, Parallel

from scipy import vstack
from scipy.sparse.lil import lil_matrix
from scipy.sparse.csr import csr_matrix
from scipy.sparse.base import issparse

from astropy.modeling.parameters import Parameter
from datetime import datetime
from pprint import pprint

from PermutationBasedIndex import PBINearestNeighbors
from PermutationBasedIndex.pivotSelection import reference_set_selection, kMedoids, kmeans,random_select_pivot,birch,kmedoidwv,reference_set_selection_wv,reference_set_selection_ordered,miniBatchkmeans,affinityPropagation,aff_prop

from locality_sensitive_hashing import LSHTransformer, LSHIINearestNeighbors, InvertedIndexNearestNeighbors,BM25NearestNeighbors
 
def encode_dataframe_content(dataframe_contenti, encoding):
    return dataframe_contenti.encode(encoding)    

def parameters_gridlist_dataframe(parameters_dict):
    
    '''
        parameters_dict is a (sklearn) Pipeline parameters dict 
        
        returns a (pandas) Dataframe containing a list of parameters' grid from parameters_dict. 
    '''
    grid_list = list(ParameterGrid(parameters_dict))

    df = pd.DataFrame(grid_list)

#     print(grid_list)    
#     print(df.to_dict("records"))
#     print(df.describe)
#     exit()
    
    return df

def h5_results_filename(dataset_name,result_type,dataframe_pos):
    h5_file_path = "%s_%s_%d_results.h5"%(dataset_name,result_type,dataframe_pos)
    return h5_file_path

def sparse_matrix_to_hdf(sparse_matrix,name_to_store,hdf_file_path):
    nonzero_indices = np.nonzero(sparse_matrix!=0)
    if len(nonzero_indices[0]) == 0:
            raise Exception("can't store empty sparse matrix!")
    
    if issparse(sparse_matrix):
        if sparse_matrix.__class__ is lil_matrix:
            nonzero_values = sparse_matrix.tocsr()[nonzero_indices].A1
        else:
            nonzero_values = lil_matrix(sparse_matrix).tocsr()[nonzero_indices].A1
    else:
        nonzero_values = np.array(sparse_matrix[nonzero_indices])

#     print(sparse_matrix.__class__,'=',name_to_store,sparse_matrix.shape,len(nonzero_values))
        
    matrix_dataframe = pd.DataFrame({
                               "row_indexes":nonzero_indices[0],
                               "col_indexes":nonzero_indices[1],
                               "data":nonzero_values})
    matrix_shape_series = pd.Series(sparse_matrix.shape)
    
    matrix_dataframe.to_hdf(hdf_file_path, name_to_store)
    matrix_shape_series.to_hdf(hdf_file_path, "%s_shape"%name_to_store)
    
    del nonzero_indices,nonzero_values,matrix_dataframe,matrix_shape_series

def hdf_to_sparse_matrix(name_to_load,hdf_file_path):
    matrix_dataframe = pd.read_hdf(hdf_file_path, name_to_load)
    matrix_shape_series = pd.read_hdf(hdf_file_path, "%s_shape"%name_to_load)

    col = matrix_dataframe.loc[:,'col_indexes'].values.tolist()
    row = matrix_dataframe.loc[:,'row_indexes'].values.tolist()
    data = matrix_dataframe.loc[:,'data'].values.tolist()

#     print(np.array(col).shape)
#     print(np.array(row).shape)
#     print(np.array(data).shape)
#     print(matrix_shape_series.values.tolist())

    sparse_matrix = csr_matrix((data, (row, col)), shape=matrix_shape_series.values.tolist())

    del col,row,data, matrix_dataframe, matrix_shape_series
    
    return sparse_matrix

def root_hypernym_tokenizer(doc):
    from nltk.corpus import wordnet as wn
    import re
    
    token_pattern = re.compile(r"(?u)\b\w\w+\b")
    token_list = []
    for tokeni in token_pattern.findall(doc):
        try:
#             token_list.append(tokeni)
            token_list.append(wn.synsets(tokeni)[0].root_hypernyms()[0]._name)
#             print(tokeni,'->',wn.synsets(tokeni)[0].root_hypernyms()[0]._pos,wn.synsets(tokeni)[0].root_hypernyms()[0]._name)
        except:
            pass
    
#     print(doc)
#     print(token_list)
#     print('==============')
    return token_list

def just_nouns_adjectives_and_verbs(doc):
    from nltk.corpus import wordnet as wn
    import re
    
    token_pattern = re.compile(r"(?u)\b\w\w+\b")
    token_list = []
    for tokeni in token_pattern.findall(doc):
        try:
            if wn.synsets(tokeni)[0].root_hypernyms()[0]._pos in ['a', 'v', 'n']:

                token_list.append(tokeni)
                
    #             print(tokeni,'->',wn.synsets(tokeni)[0].root_hypernyms()[0]._pos,wn.synsets(tokeni)[0].root_hypernyms()[0]._name)
#                 print(wn.synsets(tokeni)[0].root_hypernyms()[0]._pos,'->',tokeni)

        except:
            pass
    
#     print(doc)
#     print(token_list)
#     print('==============')
    return token_list

def just_nouns_adjectives_and_verbs_hypernyms(doc):
    from nltk.corpus import wordnet as wn
    import re
    
    token_pattern = re.compile(r"(?u)\b\w\w+\b")
    token_list = []
    for tokeni in token_pattern.findall(doc):
        try:
            if wn.synsets(tokeni)[0].root_hypernyms()[0]._pos in ['a', 'v', 'n']:

                token_list.append(wn.synsets(tokeni)[0].root_hypernyms()[0]._name)
                
    #             print(tokeni,'->',wn.synsets(tokeni)[0].root_hypernyms()[0]._pos,wn.synsets(tokeni)[0].root_hypernyms()[0]._name)
#                 print(wn.synsets(tokeni)[0].root_hypernyms()[0]._pos,'->',tokeni)

        except:
            pass
    
#     print(doc)
#     print(token_list)
#     print('==============')
    return token_list
 
def load_word_embeddings(file_path,documents):
    if not os.path.exists(file_path.replace('results.h5', 'words_in_vec.pkl')):
        t0 = time()
        model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
        words = []
        if not os.path.exists(file_path.replace('results.h5', 'words.pkl')):
            for i in range(documents.__len__()):
                text = str(documents[i])
                text = text.replace("\\n", " ")
                text = text.replace("\\xbb", " ")
                text = text.replace("\\xbf", " ")
                text = text.replace("\\xef", " ")
                text = text.replace('"', " ")
                text = text.replace("'b", " ")
                text = text.replace("\\", " ")
                text = text.replace("/", " ")
                text = text.replace(":", " ")
                text = text.replace(",", " ")
                text = text.replace(".", " ")
                text = text.replace(";", " ")
                text = text.replace("?", " ")
                text = text.replace("^", " ")
                text = text.replace("~", " ")
                text = text.replace("`", " ")
                text = text.replace("{", " ")
                text = text.replace("}", " ")
                text = text.replace("[", " ")
                text = text.replace("]", " ")
                text = text.replace("#", " ")
                text = text.replace("$", " ")
                text = text.replace("*", " ")
                text = text.replace("(", " ")
                text = text.replace(")", " ")
                text = re.sub(" +",' ',text)
                
                for word in text.split(" "):
                    if word not in words:
                        if word in model.vocab:                
                            words.append(word)
                
            
            timeCreateWordVec = time()-t0
            print("time to create a load dataset: "+str(timeCreateWordVec))
            
            with open(file_path.replace('results.h5', 'words.pkl'),'wb') as f:
                pickle.dump(words,f)
        
        w = []
        with open(file_path.replace('results.h5', 'words.pkl'),'rb') as f:
            words = pickle.load(f)
            w  = np.array(words)
            del words
        
        matrix = np.memmap(file_path.replace('results.h5', 'words_in_vec.pkl'), mode='w+', shape=(w.size,w.size),dtype=np.float16)
        for i in range(w.size):
            matrix[i,i:] = np.array(np.where(True,model.similarity(w[i:],w[i]),0),dtype=np.float16)
       
        del model     
        
        timeCreateWordVec = time()-t0
        time_word_vec_dataframe = pd.DataFrame({                   
                   'timeCreateWordVec' : [timeCreateWordVec],
                   })
        time_dataframe.to_hdf(file_path.replace('results.h5', 'time_ww.h5'), 'time_ww_dataframe')
        print("time to create a words2vec matrix: "+str(timeCreateWordVec))
        return matrix
    
def tokenize_by_parameters(documents,queries,target,dataset_name, cv_parameters_dataframe_line,cv_parameters_dataframe_line_index,encoding,dataset_encoding,use_word_embeddings):
    '''
        tokenize and store results 
    '''
    start_time = time()
    file_path = h5_results_filename(dataset_name, 'cv', cv_parameters_dataframe_line_index) #"%s_cv_%d_results.h5"%(dataset_name,cv_parameters_dataframe_line_index)
    parameters = cv_parameters_dataframe_line.to_dict()    
    
    if(use_word_embeddings == True):
        load_word_embeddings(file_path,documents)
    
    if os.path.exists(file_path):
        print(file_path,' already exists!')
    else:
        print('start:',file_path)
        pipe_to_exec = Pipeline([("cv",TfidfVectorizer(binary=True, encoding=dataset_encoding))])
        
        parameters['cv__encoding'] = encoding
    
        pipe_to_exec.set_params(**cv_parameters_dataframe_line)
        
        
        t0 = time()
        td_documents = pipe_to_exec.fit_transform(documents, None)
        documents_elapsed_time = (time()-t0) / td_documents.shape[0]
        sparse_matrix_to_hdf(td_documents,'documents',file_path)
        print('documents:',td_documents.shape)

        del td_documents
    
        t0 = time()
        td_queries = pipe_to_exec.transform(queries)
        queries_elapsed_time = (time()-t0) / td_queries.shape[0]
        sparse_matrix_to_hdf(td_queries,'queries',file_path)
        print('queries:',td_queries.shape)
        del td_queries
    
        sparse_matrix_to_hdf(target,'targets',file_path)
        
        time_dataframe = pd.DataFrame({
                   'documents_mean_time' : [documents_elapsed_time],
                   'queries_mean_time' : [queries_elapsed_time],                   
                   })
        time_dataframe.to_hdf(file_path.replace('results.h5', 'time.h5'), 'time_dataframe')
    
        del time_dataframe    
        with open(file_path.replace('results.h5', 'vocabulary.pkl'),'wb') as f:
            pickle.dump(pipe_to_exec.steps[0][1].vocabulary_,f)
        
        del pipe_to_exec
         
        print('end:',file_path, "in %4.2f s"%(time()-start_time)) 

        d = hdf_to_sparse_matrix('documents', file_path)
        
        for i in range(d.shape[0]):
            a = d[i,:].sum()
            if a ==0 :
                print(i,'/',d.shape[0])
                input('vazio!')
                print(documents[i])

        d = hdf_to_sparse_matrix('queries', file_path)
        
        for i in range(d.shape[0]):
            a = d[i,:].sum()
            if a ==0 :
                print(i,'/',d.shape[0])
                input('vazio!')
                print(queries[i])
    
    
        

def lsh_transform(dataset_name, lsht_parameters_dataframe_line, lsth_parameters_dataframe_line_index, encoding):
    indexi  = lsht_parameters_dataframe_line['input__filename_index']
    source_file_path = h5_results_filename(dataset_name, 'cv', indexi)
    file_path = h5_results_filename(dataset_name, 'lsht', lsth_parameters_dataframe_line_index)
    
    if os.path.exists(file_path) :
        print(file_path,' already exists!')
    else:    
        print(file_path,' don\'t exists!')
        print(lsht_parameters_dataframe_line) 
#        exit()
        
        pipe_to_exec = Pipeline([('lsht',LSHTransformer(n_permutations=1,n_jobs=0))])
        pipe_to_exec.set_params(**lsht_parameters_dataframe_line.drop('input__filename_index'))
        
        print(lsht_parameters_dataframe_line.drop('input__filename_index'))
        
        
        d = hdf_to_sparse_matrix('documents', source_file_path)

        d_line,d_time = pipe_to_exec.fit_transform(d, None)
        sparse_matrix_to_hdf(d_line,'documents',file_path) 
        sparse_matrix_to_hdf(d_time,'documents_time',file_path.replace('results.h5', 'time.h5'))
        print(d_line.shape, "in %f[+/-%4.2f] s"%(d_time.mean(),d_time.std()))
        
        del d,d_line,d_time
        
        q = hdf_to_sparse_matrix('queries', source_file_path)

        q_line,q_time = pipe_to_exec.transform(q)
        sparse_matrix_to_hdf(q_line,'queries',file_path)
        sparse_matrix_to_hdf(q_time,'queries_time',file_path.replace('results.h5', 'time.h5'))
        print(q_line.shape, "in %f[+/-%4.2f] s"%(q_time.mean(),q_time.std()))
        
        del q,q_line,q_time
        
        t = hdf_to_sparse_matrix('targets', source_file_path)
        sparse_matrix_to_hdf(t,'targets',file_path)
        del t

def __nearest_neighbors_search(pipe_to_exec,source_file_path,file_path):
    '''
        runs "pipe_to_exec" nearest neighbors search estimator
            
        parameters: 
        
            * source_file_path : hdf file in which input documents, queries and targets are stored
            * file_path: hdf filename where nns results will be stored
    '''
        
#     print(linei.describe)
        
    d = hdf_to_sparse_matrix('documents', source_file_path)
    pipe_to_exec.fit(d, None)
    d_mean_time = pipe_to_exec.steps[0][1].fit_time
         
    print("fitted in %f s"%(d_mean_time))
        
    del d
        
    q = hdf_to_sparse_matrix('queries', source_file_path)
    d_indices,qd_distances,q_mean_time = pipe_to_exec.transform(q)
        
#     print("mean retrieval time %f s"%(q_mean_time))
        
    time_dataframe = pd.DataFrame({
               'documents_mean_time' : [d_mean_time],
               'queries_mean_time' : [q_mean_time],
            })
        
    '''
        storing nearest neighbors search results
    '''
    time_dataframe.to_hdf(file_path.replace('results.h5', 'time.h5'), 'time_dataframe')
#    sparse_matrix_to_hdf(d_indices,'retrieved_docs',file_path)
    sparse_matrix_to_hdf(lil_matrix(qd_distances),'qd_distances',file_path)
        
    del q, d_mean_time, q_mean_time, qd_distances, time_dataframe
        
    '''
        Evaluating results in terms of Precision, Recalls and MAP.
    '''

    t = hdf_to_sparse_matrix('targets', source_file_path)
        
    retrieved_relevants = []
    for q_index in range(d_indices.shape[0]):
        q_retrieved_relevants = np.cumsum(t[q_index,d_indices[q_index,:]].A,axis=1)
        retrieved_relevants.append(q_retrieved_relevants)
        
    retrieved_relevants = vstack(retrieved_relevants)
    print("retrieved_relevants",len(retrieved_relevants))        
    '''
        broadcasting
    '''        
    approachi_recalls = np.divide(retrieved_relevants,np.matrix(t.sum(axis=1)))
    ranking_sum = np.multiply(np.ones(retrieved_relevants.shape),np.matrix(range(1,retrieved_relevants.shape[1]+1)))
    approachi_precisions = np.divide(retrieved_relevants,ranking_sum)
        
    average_precision = np.zeros((d_indices.shape[0],1))
    for q_index in range(d_indices.shape[0]):
        relevants_precision = np.multiply(approachi_precisions[q_index,:],t[q_index,d_indices[q_index,:]].A)
        average_precision[q_index,0] = relevants_precision.mean(axis=1)
#         print(q_index,'.MAP =',average_precision[q_index,0])    

#     print(t.sum(axis=1))
#     print(retrieved_relevants)
    del d_indices, retrieved_relevants

#     print("MAP=",average_precision.mean(),average_precision.std(),'precision.sum=',average_precision.sum())
#     print("recalls.sum = ",approachi_recalls.sum(),'| mean = ',approachi_recalls.sum()/(approachi_recalls.shape[0]*approachi_recalls.shape[1]))
        
    for to_store,to_store_name in [(approachi_precisions,'precisions'),(approachi_recalls,'recalls'),(average_precision,'average_precisions')]:
        if not issparse(to_store):
            to_store = csr_matrix(to_store)
        sparse_matrix_to_hdf(to_store,to_store_name,file_path.replace('results','results_evaluation'))
        
        del to_store

def lsh_nearest_neighbors_search(dataset_name, lshnns_parameters_dataframe_line, lshnns_parameters_dataframe_line_index, encoding):
    indexi = lshnns_parameters_dataframe_line['input__filename_index']
    source_file_path = h5_results_filename(dataset_name, 'lsht', indexi)
    file_path = h5_results_filename(dataset_name, 'lshnns', lshnns_parameters_dataframe_line_index)
    
    print(lshnns_parameters_dataframe_line)

    if os.path.exists(file_path):
#     if os.path.exists(file_path):
        print(file_path,' already exists!')
    else:    
        print(file_path,' don\'t exists!')
        print(lshnns_parameters_dataframe_line)
#        exit() 
    
        pipe_to_exec = Pipeline([('lshnns',LSHIINearestNeighbors(n_neighbors=2))])
        pipe_to_exec.set_params(**lshnns_parameters_dataframe_line.drop('input__filename_index'))
        
        __nearest_neighbors_search(pipe_to_exec, source_file_path, file_path)
        

def pbinearest_neighbors_search(dataset_name, nns_parameters_dataframe_line, nns_parameters_dataframe_line_index, encoding):
    indexi = nns_parameters_dataframe_line['input__filename_index']
    nns_parameters_dataframe_line["pbinns__pivot_parameters"] = load_pivot_selection_parameters(nns_parameters_dataframe_line["pbinns__pivot_parameters"])
    
    if "pbinns__using_lsh" in nns_parameters_dataframe_line and nns_parameters_dataframe_line["pbinns__using_lsh"]:        
        
        source_file_path = h5_results_filename(dataset_name, 'lsht', indexi)
        technique_name = "lsh_pbinns"
    else:
        source_file_path = h5_results_filename(dataset_name, 'cv', indexi)
        technique_name = "pbinns"
    
    if("pbinns__pivot_parameters" in nns_parameters_dataframe_line):
        
        pivot_parameters = nns_parameters_dataframe_line["pbinns__pivot_parameters"]        
        vocabulary_file_path = dataset_name +"_cv_" + str(indexi) +"_vocabulary.pkl" 
        with open(vocabulary_file_path,'rb') as f:
            pivot_parameters["pbinns__vocabulary"] = pickle.load(f)
        
        if(nns_parameters_dataframe_line["pbinns_load_word_embeddings"]):
            matrix_wordvec = dataset_name +"_cv_" + str(indexi) +"_words_in_vec.pkl" 
            fnemmap = np.memmap(matrix_wordvec, dtype='float16', mode='r',)
            pivot_parameters["pbinns__words_in_vec"]  = np.array(fnemmap,dtype='float16')
            del fnemmap
            
            matrix_words = dataset_name +"_cv_" + str(indexi) +"_words.pkl" 
            with open(matrix_words,'rb') as f:
                pivot_parameters["pbinns__words"] = pickle.load(f)       
#         nns_parameters_dataframe_line_index["pbinns__pivot_parameters"] = pivot_parameters
       
    file_path = h5_results_filename(dataset_name, technique_name, nns_parameters_dataframe_line_index)

    if os.path.exists(file_path):
        print(file_path,' already exists!')
    else:    
        pipe_to_exec = Pipeline([(technique_name,PBINearestNeighbors(nns_parameters_dataframe_line))])
        #pipe_to_exec.set_params(**nns_parameters_dataframe_line.drop('input__filename_index'))
        
        __nearest_neighbors_search(pipe_to_exec, source_file_path, file_path)
        
def nearest_neighbors_search(dataset_name, nns_parameters_dataframe_line, nns_parameters_dataframe_line_index, encoding):
    indexi = nns_parameters_dataframe_line['input__filename_index']
    source_file_path = h5_results_filename(dataset_name, 'cv', indexi)
    file_path = h5_results_filename(dataset_name, 'nns', nns_parameters_dataframe_line_index)

#     print(nns_parameters_dataframe_line)
    
    if os.path.exists(file_path) :
        print(file_path,' already exists!')
    else:    
        pipe_to_exec = Pipeline([('nns',InvertedIndexNearestNeighbors(n_neighbors=2))])
        pipe_to_exec.set_params(**nns_parameters_dataframe_line.drop('input__filename_index'))

        __nearest_neighbors_search(pipe_to_exec, source_file_path, file_path)


def bm25_nearest_neighbors_search(dataset_name, bm25nns_parameters_dataframe_line, bm25nns_parameters_dataframe_line_index, encoding):
    indexi = bm25nns_parameters_dataframe_line['input__filename_index']
    source_file_path = h5_results_filename(dataset_name, 'cv', indexi)
    file_path = h5_results_filename(dataset_name, 'bm25nns', bm25nns_parameters_dataframe_line_index)

#     print(nns_parameters_dataframe_line)
    
    if os.path.exists(file_path) :
        print(file_path,' already exists!')
    else:    
        pipe_to_exec = Pipeline([('bm25nns',BM25NearestNeighbors(n_neighbors=bm25nns_parameters_dataframe_line["bm25nns__n_neighbors"]))])
        #pipe_to_exec.set_params(**bm25nns_parameters_dataframe_line.drop('input__filename_index'))

        __nearest_neighbors_search(pipe_to_exec, source_file_path, file_path)

def generate_or_load_parameters_grids(parameters_sequence,dataset_name):
    parameters_grids_list = []

    for i in range(len(parameters_sequence)):
        sufix,_parameters = parameters_sequence[i]
        dataframe_df_path = "%s_parameters_%s.h5"%(dataset_name,sufix)
        
        dataframe_df_paramaters = parameters_gridlist_dataframe(_parameters).drop_duplicates()
        
        '''
            search and add new parameters!
        '''
        if os.path.exists(dataframe_df_path):
            print(dataframe_df_path,'exists! merging with new values!')

            current_df = pd.read_hdf(dataframe_df_path, "parameters")        
            
            '''
                getting what already exists!
            '''

            intersection = pd.merge(current_df,dataframe_df_paramaters)
            current_df.to_csv(dataframe_df_path.replace('.h5','.csv'))
            
            if intersection.shape[0] > 0:
                for _,rowi in dataframe_df_paramaters.iterrows():
                    inter_i = pd.merge(pd.DataFrame([rowi]),intersection)
                    
                    if inter_i.shape[0] == 0:
                        temp = pd.DataFrame([rowi])
                        temp.index =[current_df.shape[0]]
                        current_df = current_df.append(temp)
            else:
                temp = dataframe_df_paramaters
                temp.index = list(range(current_df.shape[0],current_df.shape[0] + dataframe_df_paramaters.shape[0]))
#                 print(temp.index)
                current_df = current_df.append(temp)
        else:
            current_df = dataframe_df_paramaters

        current_df.to_hdf(dataframe_df_path, "parameters")
        
        '''
            preserving original index! 
        '''
        current_df.loc[list(current_df.index),'%s_file_index'%(sufix)] = list(current_df.index)
        parameters_grids_list.append(pd.merge(current_df,dataframe_df_paramaters))
        parameters_grids_list[-1] = parameters_grids_list[-1].set_index('%s_file_index'%(sufix),drop=True)
                
        if i  < len(parameters_sequence) - 1:
            parameters_sequence[i+1][1]['input__filename_index'] = parameters_grids_list[-1].index
    
    return parameters_grids_list

def load_pivot_selection_parameters(json_parameters):
    parameters_pivot_selection = json.loads(json_parameters)
    method_name =  parameters_pivot_selection["pivot_selection_function"] 
    possibles = globals().copy()
    possibles.update(locals())
    method = possibles.get(method_name)
    if not method:
         raise NotImplementedError("Method %s not implemented" % method_name)
    
    parameters_pivot_selection["pivot_selection_function"] = method
    
    return parameters_pivot_selection

def print_pbi(cv_df_paramaters, pbinns_df_paramaters,dataset_name,documents_count,queries_count):
    
    today = datetime.now()
    today = today.strftime('%Y-%m-%d_%H-%M-%S_')
    
    b = pd.merge(cv_df_paramaters, pbinns_df_paramaters, how='inner', left_index=True, right_on=['input__filename_index',],)
    
    for rowi in b.iterrows():
        cv_index = rowi[1]['input__filename_index']
        pbinns_index = rowi[0]
        
        if "pbinns__using_lsh" in rowi[1] and rowi[1]["pbinns__using_lsh"]:              
            technique_name = "lsh_pbinns"
        else:            
            technique_name = "pbinns"
   
        cv_file_path = h5_results_filename(dataset_name, 'cv', cv_index).replace('results','time')
        pbinns_file_path = h5_results_filename(dataset_name, technique_name, pbinns_index).replace('results','results_evaluation')
        pbinns_time_file_path = h5_results_filename(dataset_name, technique_name, pbinns_index).replace('results','time')
  
        approach_precisions = hdf_to_sparse_matrix('precisions', pbinns_file_path)
        approach_recalls = hdf_to_sparse_matrix('recalls', pbinns_file_path)
        average_precision = hdf_to_sparse_matrix('average_precisions', pbinns_file_path).todense()
          
        b.loc[pbinns_index,'MAP'] = average_precision.mean()
        b.loc[pbinns_index,'MAP_std'] = average_precision.std()
        b.loc[pbinns_index,'recall_mean'] = approach_recalls[:,-1].todense().mean()
        b.loc[pbinns_index,'recall_std'] = approach_recalls[:,-1].todense().std()
        b.loc[pbinns_index,'precision_mean'] = approach_precisions[:,-1].todense().mean()
        b.loc[pbinns_index,'precision_std'] = approach_precisions[:,-1].todense().std()
          
        del approach_precisions, approach_recalls, average_precision
  
        b.loc[pbinns_index,'documents_count'] = documents_count
        b.loc[pbinns_index,'queries_count'] = queries_count
          
        with open(cv_file_path.replace('time.h5', 'vocabulary.pkl'),'rb') as f:
            b.loc[pbinns_index,'vocabulary_size'] = len(pickle.load(f))
         
        b.loc[pbinns_index,'indexing_mean_time'] = 0
        b.loc[pbinns_index,'querying_mean_time'] = 0
 
        cv_time_dataframe = pd.read_hdf(cv_file_path, 'time_dataframe')
        b.loc[pbinns_index,'cv_documents_mean_time'] = cv_time_dataframe.loc[0,'documents_mean_time'] 
        b.loc[pbinns_index,'cv_queries_mean_time'] = cv_time_dataframe.loc[0,'queries_mean_time']
          
        b.loc[pbinns_index,'indexing_mean_time'] += b.loc[pbinns_index,'cv_documents_mean_time'] 
        b.loc[pbinns_index,'querying_mean_time'] += b.loc[pbinns_index,'cv_queries_mean_time']
        del cv_time_dataframe 
  
        pbinns_time_dataframe = pd.read_hdf(pbinns_time_file_path, 'time_dataframe')
        b.loc[pbinns_index,'pbinns_documents_mean_time'] = pbinns_time_dataframe.loc[0,'documents_mean_time'] 
        b.loc[pbinns_index,'pbinns_queries_mean_time'] = pbinns_time_dataframe.loc[0,'queries_mean_time']
          
        b.loc[pbinns_index,'indexing_mean_time'] += b.loc[pbinns_index,'pbinns_documents_mean_time'] 
        b.loc[pbinns_index,'querying_mean_time'] += b.loc[pbinns_index,'pbinns_queries_mean_time']
        del pbinns_time_dataframe 
  
        print('pbinns:')
        print(rowi)
        print("MAP = %4.2f[+-%4.2f]"%(b.loc[pbinns_index,'MAP'],b.loc[pbinns_index,'MAP_std']))
        print("recall = %4.2f[+-%4.2f]"%(b.loc[pbinns_index,'recall_mean'],b.loc[pbinns_index,'recall_std']))
        print("index time = %4.4f"%(b.loc[pbinns_index,'cv_documents_mean_time']+b.loc[pbinns_index,'pbinns_documents_mean_time']))
        print("query time = %4.4f"%(b.loc[pbinns_index,'cv_queries_mean_time']+b.loc[pbinns_index,'pbinns_queries_mean_time']))
  
    b.to_csv('%s%s_pbi_results.csv'%(today,dataset_name),sep='\t')


def print_technique(cv_index,pbinns_index,dataset_name):
    cv_file_path = h5_results_filename(dataset_name, 'cv', cv_index).replace('results','time')
    pbinns_file_path = h5_results_filename(dataset_name, 'pbinns', pbinns_index).replace('results','results_evaluation')
    pbinns_time_file_path = h5_results_filename(dataset_name, 'pbinns', pbinns_index).replace('results','time')
    
    approach_precisions = hdf_to_sparse_matrix('precisions', pbinns_file_path)
    approach_recalls = hdf_to_sparse_matrix('recalls', pbinns_file_path)
    average_precision = hdf_to_sparse_matrix('average_precisions', pbinns_file_path).todense()
    
    cv_time_dataframe = pd.read_hdf(cv_file_path, 'time_dataframe')
    pbinns_time_dataframe = pd.read_hdf(pbinns_time_file_path, 'time_dataframe')
        
    print('pbinns:')
    print("MAP = %4.2f[+-%4.2f]"%(average_precision.mean(),average_precision.std()))
    print("recall = %4.2f[+-%4.2f]"%(approach_recalls[:,-1].todense().mean(),approach_recalls[:,-1].todense().std()))
    print("index time = %4.4f"%(cv_time_dataframe.loc[0,'documents_mean_time']+pbinns_time_dataframe.loc[0,'documents_mean_time']))
    print("query time = %4.4f"%(cv_time_dataframe.loc[0,'queries_mean_time']+pbinns_time_dataframe.loc[0,'queries_mean_time']))
