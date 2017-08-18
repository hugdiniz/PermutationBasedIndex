import numpy as np

from time import time
import os
import gc
import sys
import pickle

from datasets.extractors import meter_extractor, \
short_plagiarised_answers_extractor, pan_plagiarism_corpus_2011_extractor, \
pan_plagiarism_corpus_2010_extractor

  
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle

    
   

if __name__ == "__main__":
    
    '''
        experiments Variables
    '''
    corpus_name ="psa"
#   corpus_name ="pan11"
#   corpus_name ="pan10"
    
    if len(sys.argv) > 2:        
        permutation_count_list = [int(sys.argv[1])]
        part_number = int(sys.argv[2])
    else:
        permutation_count_list = [100]
        part_number = 0
        
        
    path_file = "results_%s/part_%d"%(corpus_name,part_number)    
    if not os.path.exists(path_file):
        os.makedirs(path_file)

    '''
        dataset extraction
    '''

    if corpus_name == "meter" or corpus_name == "psa":
        if corpus_name == "meter":
            dataset_documents,dataset_target,_,_,dataset_encoding = meter_extractor.load_as_pairs()
            nonzero_indexes = np.argwhere(dataset_target)
        elif corpus_name == "psa":
            corpus_name, (dataset_documents,dataset_target,_,_,dataset_encoding) = "psa", short_plagiarised_answers_extractor.load_as_pairs()
            nonzero_indexes = range(dataset_documents.shape[0])

#         print(dataset_documents.shape)
        
        documents,queries = [],[] 
        for i in nonzero_indexes:
            queries.append(dataset_documents[i,0])
            documents.append(dataset_documents[i,1])
            if corpus_name == "meter":
                queries[-1] = queries[-1].flatten()[0,0] 
                documents[-1] = documents[-1].flatten()[0,0] 
            
        del dataset_documents,dataset_target,dataset_encoding    
    else:
        if corpus_name == "pan11" or corpus_name == "pan10":
            if corpus_name == "pan11":
                queries_, doc_index_, dataset_target, dataset_encoding = pan_plagiarism_corpus_2011_extractor.load_as_ir_task(language_filter = 'EN')
            else:
                queries_, doc_index_, dataset_target, dataset_encoding = pan_plagiarism_corpus_2010_extractor.load_as_ir_task(language_filter = 'EN')
            nonzero_indexes = np.argwhere(dataset_target)
#             print(dataset_target)
#             print(nonzero_indexes)
#             print(dataset_target.shape,len(nonzero_indexes))

            documents,queries = [],[] 
            for nzi in nonzero_indexes[:1000]:
#                 print(nzi)
                queries.append(queries_.loc[nzi[0],'content'])
                documents.append(doc_index_.loc[nzi[1],'content'])
            
            del queries_, doc_index_,dataset_target,dataset_encoding
    
#     print(queries[-1])
#     print(len(queries),len(documents))
#     print(documents[0])
#     exit()
    
    
    '''
        using scikit-learn : tokenization
    '''    
    vectorizer = CountVectorizer(binary=True,min_df=1,ngram_range=(1,1))
    all_fingerprints = vectorizer.fit_transform(queries+documents, None).T
#     vocabulary_indexes = [di for di in vectorizer.vocabulary_.values()]
    del queries, documents,vectorizer
    gc.collect()

    print("all_fingerprints: ",all_fingerprints.shape)
    
    fname = "true_jaccard_sim_%s.pkl"%(corpus_name)
    if(os.path.isfile(fname)):
        print("Reading true_jaccard_similarity")
        with open(fname,'rb') as f:
            true_jaccard_sim = pickle.load(f)
        
    else:
        true_jaccard_sim = pairwise_distances(np.vstack([i*all_fingerprints[i,:].toarray() for i in range(all_fingerprints.shape[0])]).T,metric='jaccard', n_jobs=1)
        with open(fname,'wb') as f:
            print("Writing true_jaccard_similarity")
            pickle.dump(true_jaccard_sim, f)
    
    true_jaccard_sim_mean, true_jaccard_sim_std = true_jaccard_sim.mean(), true_jaccard_sim.std()
    print('true_jaccard_sim (mean,std) =(',true_jaccard_sim_mean,',',true_jaccard_sim_std,')')

    results = {}
    '''
        using scikit-learn : permutation
            each permutation has one term-document matrix
    '''    
    permutation_repetition = 100