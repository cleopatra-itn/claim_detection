## Features: Bert Word Embeddings
## ## Models: KD-Tree Search

import sys, os, io, time, string

from sklearn.neighbors import KDTree
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import decomposition, ensemble, tree

from nltk.corpus import stopwords
from elasticsearch import Elasticsearch


import json, random, math, re, string
import pandas as pd
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer, SentencesDataset, losses
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.readers import TripletReader
from sentence_transformers.losses.TripletLoss import TripletDistanceMetric

import argparse

parser = argparse.ArgumentParser(description='Finetune with Bert Sent Embeddings')
parser.add_argument('--proc', type=int, default=0, help="0,1")
parser.add_argument('--negs', type=int, default=5, help="1-10")
parser.add_argument('--bert_type', type=int, default=0, help="Bert Type, Base, Large, Roberta, etc...")

args = parser.parse_args()

my_loc = os.path.dirname(__file__)


def query_claim_dict(pairs):
    dct = {}
    for i in range(len(pairs)):
        q_id, _, cl_ind, _ = pairs[i].split()
        if q_id not in dct:
            dct[q_id] = [int(cl_ind)]
        else:
            dct[q_id].append(int(cl_ind))
        
    return dct


def id_to_index_dict(ids):
    temp_dict = {}
    cnt = 0
    for idx in ids:
        temp_dict[idx] = cnt
        cnt += 1

    return temp_dict

def get_triples(id_to_indx, raw_pairs, q_txt_dict, q_claim_dict, claim_txt_dict, title_txt_dict, cos_sim, phase):

    triples = []
    for i in range(len(raw_pairs)):
        q_id, _, cl_ind, _ = raw_pairs[i].split()
        query = q_txt_dict[q_id].strip()
        claim = claim_txt_dict[cl_ind].strip()
        title = title_txt_dict[cl_ind].strip()

        true_ids = q_claim_dict[q_id]

        sims = cos_sim[id_to_indx[q_id],:]*-1
        sims[true_ids] += -100
    
        if phase == 'train':
            top_sort_inds = np.argsort(sims)[len(true_ids):args.negs]
        else:
            top_sort_inds = np.argsort(sims)[len(true_ids):3]

        for c in top_sort_inds:
            claim_neg = claim_txt_dict[str(c)].strip()
            title_neg = title_txt_dict[str(c)].strip()
            if phase == 'train':
                triples.append([query, claim, claim_neg])
                triples.append([query, title, title_neg])
                # triples.append([query, title+" "+claim, title_neg+" "+claim_neg])
            else:
                triples.append([query, claim, claim_neg])
                # triples.append([query, title+" "+claim, title_neg+" "+claim_neg])
        
    return triples


def get_proc_text(data, text_type):
    txt_dict = {}
    for idx in data:
        text = data[idx][text_type]

        text = [word for word in text if not re.search(r'<(/?)[a-z]+>', word)]

        proc_text = ""
        for word in text:
            proc_text += word if word in [',', '.'] else " "+word

        txt_dict[idx] = proc_text
    
    return txt_dict



files = ['_raw_text', '_proc_text']
fname = files[args.proc]

emb_list = ['bert-base-nli-mean-tokens', 'bert-base-nli-cls-token', 'bert-base-nli-max-tokens',
                'bert-large-nli-mean-tokens', 'bert-large-nli-cls-token', 'bert-large-nli-max-tokens',
                'roberta-base-nli-mean-tokens', 'roberta-large-nli-mean-tokens', 'distilbert-base-nli-mean-tokens',
                'bert-base-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens', 'roberta-base-nli-stsb-mean-tokens', 
                'roberta-large-nli-stsb-mean-tokens', 'distilbert-base-nli-stsb-mean-tokens', 'distiluse-base-multilingual-cased']

emb_type = emb_list[args.bert_type]

emb_data = json.load(open(my_loc+'/bert_embs/'+emb_type+fname+'.json', 'r')) 
tr_embs = emb_data['train']['embs']
val_embs = emb_data['val']['embs']
cl_embs = emb_data['claims']['embs']
tr_ids = emb_data['train']['id']
val_ids = emb_data['val']['id']

train_data = json.load(open(my_loc+'/proc_data/train.json', 'r'))
val_data = json.load(open(my_loc+'/proc_data/val.json', 'r'))
claim_data = json.load(open(my_loc+'/proc_data/claim_dict.json', 'r'))

## Dict with text matched to tweet IDs
if 'raw' in fname:
    tr_txt_dict = {idx: train_data[idx]['text'] for idx in train_data}
    val_txt_dict = {idx: val_data[idx]['text'] for idx in val_data}
    claim_txt_dict = {idx: claim_data[idx]['claim'] for idx in claim_data}
    title_txt_dict = {idx: claim_data[idx]['title'] for idx in claim_data}
else:
    tr_txt_dict = get_proc_text(train_data, 'wiki_proc')
    val_txt_dict = get_proc_text(val_data, 'wiki_proc')
    claim_txt_dict = get_proc_text(claim_data, 'claim_proc')
    title_txt_dict = get_proc_text(claim_data, 'title_proc')


tr_pairs = open('data/train/tweet-vclaim-pairs.qrels','r', encoding='utf-8').readlines()
val_pairs = open('data/dev/tweet-vclaim-pairs.qrels', 'r', encoding='utf-8').readlines()

## Dict with claim IDs matched to tweet IDs
tr_claim_dict = query_claim_dict(tr_pairs)
val_claim_dict = query_claim_dict(val_pairs)

tr_id_to_index_dict = id_to_index_dict(tr_ids)
val_id_to_index_dict = id_to_index_dict(val_ids)

## Cosine similarity 
cos_sim_tr = cosine_similarity(tr_embs, cl_embs)
cos_sim_val = cosine_similarity(val_embs, cl_embs)


train_trips = get_triples(tr_id_to_index_dict, tr_pairs, tr_txt_dict, tr_claim_dict, claim_txt_dict, title_txt_dict, cos_sim_tr, 'train')
val_trips = get_triples(val_id_to_index_dict, val_pairs, val_txt_dict, val_claim_dict, claim_txt_dict, title_txt_dict, cos_sim_val, 'val')



with open(my_loc+'/proc_data/train-triples.csv', 'w', encoding='utf-8') as f:
    for trs in train_trips:
        f.write('%s\t%s\t%s\n'%(trs[0], trs[1], trs[2]))


with open(my_loc+'/proc_data/val-triples.csv', 'w', encoding='utf-8') as f:
    for trs in val_trips:
        f.write('%s\t%s\t%s\n'%(trs[0], trs[1], trs[2]))



num_epochs = 2
batch_size = 8
model_save_path = my_loc+'/models/finetune_%s_%s'%(fname,emb_type)

triplet_reader = TripletReader(dataset_folder=my_loc+'/proc_data/')

model = SentenceTransformer(emb_type)
train_data = SentencesDataset(triplet_reader.get_examples('train-triples.csv'), model)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
train_loss = losses.TripletLoss(model=model, triplet_margin=1)

dev_data = SentencesDataset(examples=triplet_reader.get_examples('val-triples.csv'), model=model)
dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=batch_size)
evaluator = TripletEvaluator(dev_dataloader)

warmup_steps = math.ceil(len(train_data)*num_epochs/batch_size*0.1) #10% of train data for warm-up

model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=500,
          warmup_steps=warmup_steps,
          output_path=model_save_path)



model = SentenceTransformer(model_save_path)


