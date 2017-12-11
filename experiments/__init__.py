'''
    Created on 11/30/2017
    @author: Hugo Rebelo
'''
import os
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle
from time import time
from math import floor
import os.path
import sys
from time import time
from PermutationBasedIndex.lsh import lsh
from datasets.extractors import short_plagiarised_answers_extractor, pan_plagiarism_corpus_2010_extractor, pan_plagiarism_corpus_2011_extractor
from locality_sensitive_hashing import minmax_hashing
from sklearn.metrics import recall_score
import json
from scipy import sparse

'''
        IN -- Recieve dataset(In this case, fingerprints), document_index(index for dataset, not 'real indexes') and LSH variables(approachs , number of permutation and cell numbers for features partition approachs)
        OUT -- indexes(index for dataset) and time to execute nearest neighbors
'''    
def nearest_neighbors_LSH(dataset,doc_index,approach = minmax_hashing,permutation = 100, cells_number = 2):
    t0 = time()
    lsh_dataset,time_lsh = lsh(dataset)
    matrixDistance = pairwise_distances(lsh_dataset,metric='jaccard', n_jobs=1)
    similarity_docs =matrixDistance[doc_index,:]
    t1 = time() - t0
    return np.argsort(similarity_docs),t1
    
def gerArrayPercent(array,percentNumber):
    size = (percentNumber/100) * array.size
    return array[:size]

def getRealIndex(arrayRealIndex, index):
    return arrayRealIndex[index]


def loadPSA():
    
    paths =  [name for name in os.listdir(".") if os.path.isdir(name)]
    documents,queries = [],[]
    jsonDatas = json.load(open('query_answer_json'))
    dataset_encoding='latin1'
    tasks = ['a','b','c','d','e']
    index_queries = []
    index_queries_task = []    
    i = 0
    for jsonData in jsonDatas:
        path = jsonData['plag_type']
        file = jsonData['document']    
        with open(os.path.join(path, file),encoding=dataset_encoding) as f:
            text = ""            
            for line in f:
                text = text + line                
            queries.append(text)
            index_queries.append(i)
            index_queries_task.append(tasks.index(jsonData['task']))           
            i = i + 1
    
    
    r = np.array([1 for j in range(index_queries.__len__())])
    index_queries = np.array(index_queries)
    index_queries_task = np.array(index_queries_task)
    
    target = sparse.coo_matrix((r,(index_queries,index_queries_task)),shape=(index_queries_task.size,index_queries_task.size))
    
    for task in tasks:  
        file = 'orig_task' + task + '.txt'        #print(os.path.join("path", file))
        with open(os.path.join('source', file),encoding=dataset_encoding) as f:
            text = ""
            for line in f:
                text = text + line
            documents.append(text)    
    
  
    return documents,queries,target
            

if __name__ == '__main__':
    
    '''
        INIT -- Variables
    '''
    size_percent = 50
    dataset_name = "psa"    
    '''
        END -- Variables
    '''
    

    '''
        INIT -- dataset extraction
    '''
    
    documents,queries = [],[] 
    documents_index,queries_index = [],[] 
    if dataset_name == "psa":
        corpus_name, (queries, documents, dataset_target, dataset_encoding) = dataset_name, short_plagiarised_answers_extractor.load_as_ir_task()
    elif dataset_name == "pan10":
        corpus_name, (suspicious_info, source_info,target, dataset_encoding) = dataset_name, pan_plagiarism_corpus_2010_extractor.load_as_ir_task(allow_queries_without_relevants=False)
    elif dataset_name == "pan11":
        corpus_name, (suspicious_info, source_info,target, dataset_encoding) = dataset_name, pan_plagiarism_corpus_2011_extractor.load_as_ir_task(allow_queries_without_relevants=False)
    elif "pan10" in dataset_name and "-samples" in dataset_name:
        corpus_name, (suspicious_info, source_info,target, dataset_encoding) = dataset_name, pan_plagiarism_corpus_2010_extractor.load_sample_as_ir_task(sample_size)
  
    
    
    
    
    for nzi in nonzero_indexes[:1000]:
        documents.append(suspicious_info.loc[nzi[1],'content'])
        queries.append(source_info.loc[nzi[0],'content'])
        documents_index.append(nzi[1])
        queries_index.append(nzi[0])
    
    vectorizer = CountVectorizer(binary=True,min_df=1,ngram_range=(1,1))
    all_fingerprints = vectorizer.fit_transform(queries+documents, None).T     
    
    '''
        END -- dataset extraction
    '''
    recallNumber = []     
    for i in range(documents_index.__sizeof__()):
        similaritys_index,time = nearest_neighbors_LSH(all_fingerprints,i)
        indexs_lsh = np.array([getRealIndex(documents_index, i) for i in gerArrayPercent(similaritys_index,50)])
        pred = find(target[0,:])[1]
        recallNumber.append(recall_score(pred, indexs_lsh))
        
    print(recallNumber)
    
    
   