## Features: Word Embeddings
## ## Models: KD-Tree Search

import sys, os, io

from sklearn.neighbors import KDTree
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import decomposition, ensemble, tree

from nltk.corpus import stopwords
from elasticsearch import Elasticsearch

import json
import pandas as pd
import numpy as np
from tqdm import tqdm

import spacy
import gensim
import gensim.downloader as api

my_loc = os.path.dirname(__file__)

def get_spacy_doc_vectors(input_feats, nlp):
    vec_feat = []
    for tweet in input_feats:
        proc_tweet = nlp(tweet)
        if not proc_tweet.has_vector or proc_tweet.vector_norm == 0:
            vec_feat.append(np.zeros(300))
        else:
            tweet_vector = proc_tweet.vector
            tweet_vector = tweet_vector/proc_tweet.vector_norm
            vec_feat.append(tweet_vector)

    return np.array(vec_feat)


def get_gensim_doc_vectors(input_feats, model, size):
    vec_feat = []
    for tweet in input_feats:
        words = tweet.split()
        temp = []
        for word in words:
            if word in model.wv.vocab:
                temp.append(model.wv.get_vector(word))

        if not temp:
            temp = np.zeros(size)
        else:
            temp = np.array(temp).mean(axis=0)
            temp = temp/np.linalg.norm(temp, 2)

        vec_feat.append(temp)

    return np.array(vec_feat)


def get_tweet_data(tweet_list, w2v_type):
    twit_x, twit_id = [], []
    for id in tweet_list:
        twit_id.append(id)
        twit_x.append(" ".join(tweet_list[id][w2v_type]))

    tweetDF = pd.DataFrame()
    tweetDF['text'] = twit_x
    tweetDF['tweet_id'] = twit_id

    return twit_x, tweetDF



claim_data = json.load(open(my_loc+'/proc_data/claim_dict.json', 'r'))
val_data = json.load(open(my_loc+'/proc_data/val.json', 'r'))
test_data = json.load(open(my_loc+'/proc_data/test.json', 'r'))

claim_x, claimDF = get_tweet_data(claim_data, 'claim_clean_nostop')
title_x, titleDF = get_tweet_data(claim_data, 'title_clean_nostop')
val_x, valDF = get_tweet_data(val_data, 'wiki_clean_nostop')
test_x, testDF = get_tweet_data(test_data, 'wiki_clean_nostop')


wordvec_list = ['conceptnet-numberbatch-17-06-300','spacy','glove-twitter-25', 'glove-twitter-50', 'glove-twitter-100', 'glove-twitter-200',
                'glove-wiki-gigaword-50', 'glove-wiki-gigaword-100', 'glove-wiki-gigaword-200',
                'glove-wiki-gigaword-300', 'fasttext-wiki-news-subwords-300', 'word2vec-google-news-300']


for wordmod in wordvec_list:
    print(wordmod+"-------------------------------------------------------------------------------------\n")
    if wordmod == 'spacy':
        nlp = spacy.load('en_vectors_web_lg')
        ft_claim = get_spacy_doc_vectors(claim_x, nlp)
        ft_title = get_spacy_doc_vectors(title_x, nlp)
        ft_val = get_spacy_doc_vectors(val_x, nlp)
        ft_test = get_spacy_doc_vectors(test_x, nlp)
    else:
        wd_model = api.load(wordmod)
        sz = wordmod.split('-')[-1]
        ft_claim = get_gensim_doc_vectors(claim_x, wd_model, int(sz))
        ft_title = get_gensim_doc_vectors(title_x, wd_model, int(sz))
        ft_val = get_gensim_doc_vectors(val_x, wd_model, int(sz))
        ft_test = get_gensim_doc_vectors(test_x, wd_model, int(sz))

    ft_claim = np.add(ft_claim, ft_title)/2

    # kdtree = KDTree(ft_claim)

    # with open('my_code/file_results/w2v_res_%s.tsv'%(wordmod), 'w') as f:
    #     dists, inds = kdtree.query(ft_val, k=1000)

    #     for i in range(ft_val.shape[0]):
    #         cos_sc = cosine_similarity(np.expand_dims(ft_val[i,:],0), ft_claim[inds[i,:]]).flatten()

    #         for j in range(inds.shape[1]):
    #             f.write("%d\tQ0\t%d\t1\t%f\t%s\n"%(int(valDF['tweet_id'][i]),inds[i,j],cos_sc[j],'w2v_avg'))
    
    # os.system('python evaluate.py --scores my_code/file_results/w2v_res_%s.tsv --gold-labels data/dev/tweet-vclaim-pairs.qrels'%(wordmod))
    # print('-------------------------------------------------------------------------------------------\n')

    cos_sim_val = cosine_similarity(ft_val, ft_claim)

    with open('my_code/file_results/w2v_res_val_%s.tsv'%(wordmod), 'w') as f:
        for i in range(ft_val.shape[0]):
            # i_dist = dists[i]
            # i_dist = 1 - i_dist/max(i_dist)
            srt_vals = (np.sort(cos_sim_val[i,:]*-1)*-1)[:1000]
            srt_inds = np.argsort(cos_sim_val[i,:]*-1)[:1000]

            for j in range(len(srt_inds)):
                f.write("%d\tQ0\t%d\t1\t%f\t%s\n"%(int(valDF['tweet_id'][i]),srt_inds[j],srt_vals[j],'w2v_avg'))
    
    os.system('python evaluate.py --scores my_code/file_results/w2v_res_val_%s.tsv --gold-labels data/dev/tweet-vclaim-pairs.qrels'%(wordmod))
    print('-------------------------------------------------------------------------------------------\n')


    cos_sim_test = cosine_similarity(ft_test, ft_claim)

    with open('my_code/file_results/w2v_res_test_%s.tsv'%(wordmod), 'w') as f:
        for i in range(ft_test.shape[0]):
            # i_dist = dists[i]
            # i_dist = 1 - i_dist/max(i_dist)
            srt_vals = (np.sort(cos_sim_test[i,:]*-1)*-1)[:1000]
            srt_inds = np.argsort(cos_sim_test[i,:]*-1)[:1000]

            for j in range(len(srt_inds)):
                f.write("%d\tQ0\t%d\t1\t%f\t%s\n"%(int(testDF['tweet_id'][i]),srt_inds[j],srt_vals[j],'w2v_avg'))
    
    os.system('python evaluate.py --scores my_code/file_results/w2v_res_test_%s.tsv --gold-labels data/test/tweet-vclaim-pairs.qrels'%(wordmod))
    print('-------------------------------------------------------------------------------------------\n')




# def clear_index(es):
#     cleared = True
#     try:
#         es.indices.delete(index='vclaim')
#     except:
#         cleared = False
#     return cleared

# for feat in ft_val:
#     query =  {
#             'script_score': {
#             'query': {'match_all': {}},
#             'script': {
#                 "source": "cosineSimilarity(params.query_vector, doc[\'vector\']) + 1.0",
#                 "params": {"query_vector": feat.tolist()}
#             }
#         }
#     }
#     query_body = {'size': 10,'query': query}

#     # res = es.search(index='vclaim', body = query)
#     es.search(body=query_body, index = 'vclaim')

# es = Elasticsearch([{'host':'localhost','port':9200}])
# clear_index(es)
# es.indices.delete(index='vclaim', ignore=[400, 404])

# nlp = spacy.load('en_vectors_web_lg')
# ft_claim = get_spacy_doc_vectors(claim_x, nlp)
# ft_train = get_spacy_doc_vectors(train_x, nlp)
# ft_val = get_spacy_doc_vectors(val_x, nlp)

# mapping = {"settings": {
#             "number_of_shards": 1,
#             "number_of_replicas": 0
#         },
#         "mappings": {
#             "properties": {
#                 "id": {
#                     "type": "text" # formerly "string"
#                 },
#                 "claim": {
#                     "type": "text"
#                 },
#                 "title": {
#                     "type": "text"
#                 },
#                 "vector": {
#                     "type": "dense_vector",
#                     "dims": 300
#                 }
#         }
#     }
# }

# es.indices.create(index='vlcaim', body=mapping)

# for id in tqdm(claim_data, total=len(claim_data)):
#     cl_dict = {'id':id, 'claim': claim_data[id]['claim'],
#                 'title':claim_data[id]['title'],'vector':ft_claim[int(id),:].tolist()}


#     es.create(index='vclaim', body=cl_dict)
