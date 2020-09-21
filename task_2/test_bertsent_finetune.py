
## Features: Bert Word Embeddings
## ## Models: KD-Tree Search

import sys, os, io, time

from sklearn.neighbors import KDTree
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import decomposition, ensemble, tree

from nltk.corpus import stopwords

import json, random, math, re
import pandas as pd
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer, SentencesDataset, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import STSBenchmarkDataReader

import argparse

parser = argparse.ArgumentParser(description='Finetune with Bert Sent Embeddings')
parser.add_argument('--proc', type=int, default=0, help="0,1")
parser.add_argument('--bert_type', type=int, default=0, help="Bert Type, Base, Large, Roberta, etc...")

args = parser.parse_args()

my_loc = os.path.dirname(__file__)


files = ['_raw_text', '_proc_text']
fname = files[args.proc]

emb_list = ['bert-base-nli-mean-tokens', 'bert-base-nli-cls-token', 'bert-base-nli-max-tokens',
                'bert-large-nli-mean-tokens', 'bert-large-nli-cls-token', 'bert-large-nli-max-tokens',
                'roberta-base-nli-mean-tokens', 'roberta-large-nli-mean-tokens', 'distilbert-base-nli-mean-tokens',
                'bert-base-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens', 'roberta-base-nli-stsb-mean-tokens', 
                'roberta-large-nli-stsb-mean-tokens', 'distilbert-base-nli-stsb-mean-tokens', 'distiluse-base-multilingual-cased']


emb_type = emb_list[args.bert_type]


model_path = my_loc+'/models/finetune_%s_%s'%(fname,emb_type)
model = SentenceTransformer(model_path).eval()


val_data = json.load(open(my_loc+'/proc_data/val.json', 'r'))
claim_data = json.load(open(my_loc+'/proc_data/claim_dict.json', 'r'))

data_type = {'val': val_data, 'claims': claim_data}

if 'raw' in fname:
    embed_dict = { 'val': {'id': [], 'embs': []},
                        'test': {'id': [], 'embs': []},
                        'claims': {'id': [], 'embs': [], 'embs2': [], 'embs3':[]}}
    for phase in data_type:
        corpus = []
        corpus2 = []
        corpus3 = []
        data = data_type[phase]

        if phase == 'claims':
            for idx in data:
                claim = data[idx]['claim']
                title = data[idx]['title']

                corpus.append(claim.strip())
                corpus2.append(title.strip())
                corpus3.append(title.strip()+" "+claim.strip())

                embed_dict[phase]['id'].append(idx)

        else:
            for idx in data:
                text = data[idx]['text']
                corpus.append(text.strip())

                embed_dict[phase]['id'].append(idx)
            
        
        if phase == 'claims':
            embeddings1 = model.encode(corpus, batch_size=64)
            embeddings2 = model.encode(corpus2, batch_size=64)
            embeddings3 = model.encode(corpus3, batch_size=64)

            for emb1, emb2 in zip(embeddings1, embeddings2):
                embed_dict[phase]['embs'].append(emb1.tolist())
                embed_dict[phase]['embs2'].append(((emb1+emb2)/2).tolist())
            for emb in embeddings3:
                embed_dict[phase]['embs3'].append(emb.tolist())
        else:
            embeddings = model.encode(corpus, batch_size=64)
            for emb in embeddings:
                embed_dict[phase]['embs'].append(emb.tolist())

    json.dump(embed_dict, open(my_loc+'/bert_embs/test_%s_%s.json'%(fname,emb_type), 'w', encoding='utf-8'))

else:
    embed_dict = { 'val': {'id': [], 'embs': []},
                        'test': {'id': [], 'embs': []},
                        'claims': {'id': [], 'embs': [], 'embs2': [], 'embs3':[]}}
    for phase in data_type:
        corpus = []
        corpus2 = []
        corpus3 = []
        data = data_type[phase]

        if phase == 'claims':
            for idx in data:
                claim = data[idx]['claim_proc']
                title = data[idx]['title_proc']

                claim = [word for word in claim if not re.search(r'<(/?)[a-z]+>', word)]
                title = [word for word in title if not re.search(r'<(/?)[a-z]+>', word)]

                claim_text = ""
                for word in claim:
                    claim_text += word if word in [',', '.'] else " "+word

                title_text = ""
                for word in title:
                    title_text += word if word in [',', '.'] else " "+word

                corpus.append(claim_text.strip())
                corpus2.append(title_text.strip())
                corpus3.append(title_text.strip()+" "+claim_text.strip())

                embed_dict[phase]['id'].append(idx)

        else:
            for idx in data:
                proc_text = data[idx]['wiki_proc']
                proc_text = [word for word in proc_text if not re.search(r'<(/?)[a-z]+>', word)]

                text = ""
                for word in proc_text:
                    text += word if word in [',', '.'] else " "+word

                corpus.append(text.strip())

                embed_dict[phase]['id'].append(idx)
        
        
        if phase == 'claims':
            embeddings1 = model.encode(corpus, batch_size=64)
            embeddings2 = model.encode(corpus2, batch_size=64)
            embeddings3 = model.encode(corpus3, batch_size=64)

            for emb1, emb2 in zip(embeddings1, embeddings2):
                embed_dict[phase]['embs'].append(emb1.tolist())
                embed_dict[phase]['embs2'].append(((emb1+emb2)/2).tolist())
            for emb in embeddings3:
                embed_dict[phase]['embs3'].append(emb.tolist())
        else:
            embeddings = model.encode(corpus, batch_size=64)
            for emb in embeddings:
                embed_dict[phase]['embs'].append(emb.tolist())

    json.dump(embed_dict, open(my_loc+'/bert_embs/test_%s_%s.json'%(fname,emb_type), 'w', encoding='utf-8'))


data = json.load(open(my_loc+'/bert_embs/test_%s_%s.json'%(fname,emb_type), 'r', encoding='utf-8')) 
claim_data = data['claims']
val_data = data['val']
val_ids = val_data['id']

for clm_type in ['embs', 'embs2', 'embs3']:
    print('----------------------------------------------------------------------------------------\n')
    ft_claim = np.array(claim_data[clm_type])

    ft_val = np.array(val_data['embs'])

    kdtree = KDTree(ft_claim)

    with open('my_code/file_results/bertsent_finetune_res_%s_%s.tsv'%(fname,emb_type), 'w') as f:
        dists, inds = kdtree.query(ft_val, k=1000)

        for i in range(ft_val.shape[0]):
            i_dist = dists[i]
            i_dist = 1 - i_dist/max(i_dist)

            for j in range(inds.shape[1]):
                f.write("%d\tQ0\t%d\t1\t%f\t%s\n"%(int(val_data['id'][i]),inds[i,j],i_dist[j],'bert_word'))

    os.system('python evaluate.py --scores my_code/file_results/bertsent_finetune_res_%s_%s.tsv --gold-labels data/dev/tweet-vclaim-pairs.qrels'%(fname,emb_type))
    print('-------------------------------------------------------------------------------------------\n')