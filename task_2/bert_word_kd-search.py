## Features: Bert Word Embeddings
## ## Models: KD-Tree Search

import sys, os, io, time

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
from gensim.models.fasttext import FastText

import argparse

parser = argparse.ArgumentParser(description='KD-Tree search with Bert Word Embeddings')
parser.add_argument('--bert_type', type=int, default=0, help="0,1,2,3")

args = parser.parse_args()

my_loc = os.path.dirname(__file__)



files = ['bert-base-uncased_raw_text', 'bert-base-uncased_proc_text', 
          'bert-large-uncased_raw_text', 'bert-large-uncased_proc_text']

fname = files[args.bert_type] 

data = json.load(open(my_loc+'/bert_embs/'+fname+'.json', 'r')) 
claim_data = data['claims']
val_data = data['val']

emb_list = ['sent_word_catavg', 'sent_word_catavg_wostop', 'sent_word_sumavg',
            'sent_word_sumavg_wostop', 'sent_emb_2_last', 'sent_emb_2_last_wostop',
            'sent_emb_last', 'sent_emb_last_wostop', 'pooled_out']

for emb_type in emb_list:
    print(fname+'-------'+emb_type+'---------------------------------------------------------------------\n')
    ft_claim = np.array(claim_data[emb_type+'2'])

    ft_val = np.array(val_data[emb_type])

    kdtree = KDTree(ft_claim)

    with open('my_code/file_results/bert_res_%s_%s.tsv'%(fname, emb_type), 'w') as f:
        dists, inds = kdtree.query(ft_val, k=1000)

        for i in range(ft_val.shape[0]):
            cos_sc = cosine_similarity(np.expand_dims(ft_val[i,:],0), ft_claim[inds[i,:]]).flatten()

            for j in range(inds.shape[1]):
                f.write("%d\tQ0\t%d\t1\t%f\t%s\n"%(int(val_data['id'][i]),inds[i,j],cos_sc[j],'bert_word'))
    
    os.system('python evaluate.py --scores my_code/file_results/bert_res_%s_%s.tsv --gold-labels data/dev/tweet-vclaim-pairs.qrels'%(fname, emb_type))
    print('-------------------------------------------------------------------------------------------\n')



