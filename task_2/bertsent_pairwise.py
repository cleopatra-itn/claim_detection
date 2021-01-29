## Features: Bert Word Embeddings
## ## Models: KD-Tree Search

import sys, os, io, time

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
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import STSBenchmarkDataReader

import argparse

parser = argparse.ArgumentParser(description='Finetune with Bert Sent Embeddings')
parser.add_argument('--proc', type=int, default=0, help="0,1")
parser.add_argument('--neg_sample', type=int, default=2, help="1,2,3,4")

args = parser.parse_args()

my_loc = os.path.dirname(__file__)

def query_claim_dict(pairs):
    dct = {}
    for i in range(len(pairs)):
        q_id, _, cl_ind, _ = pairs[i].split()
        dct[q_id] = [int(cl_ind)] if q_id not in dct else dct[q_id].append(int(cl_ind))
        
    return dct


def get_pospair_feats(raw_pairs, q_txt_dict, claim_txt_dict, title_txt_dict, phase):

    pairs = []
    for i in range(len(raw_pairs)):
        q_id, _, cl_ind, _ = raw_pairs[i].split()
        query = q_txt_dict[q_id]
        claim = claim_txt_dict[cl_ind]
        title = title_txt_dict[cl_ind]
        if phase == 'train':
            # pairs.append([query.strip(), claim.strip()])
            # pairs.append([query.strip(), title.strip()])
            # pairs.append([claim.strip(), title.strip()])
            pairs.append([query.strip(), title.strip()+" "+claim.strip()])
        else:
            # pairs.append([query.strip(), claim.strip()])
            pairs.append([query.strip(), title.strip()+" "+claim.strip()])
        
    return pairs



def get_negpair_feats(ids, q_claim_dict, tr_txt_dict, claim_txt_dict, title_txt_dict, cos_sim, n_sample, phase):

    pairs = []
    cnt = 0
    for idx in ids:
        query = tr_txt_dict[idx]
        true_ids = q_claim_dict[idx]

        sims = cos_sim[cnt,:]*-1
        sims[true_ids] += -100
    
        top_sort_inds = np.argsort(sims)[1:6]

        ch = random.sample(list(top_sort_inds), n_sample)

        for c in ch:
            claim = claim_txt_dict[str(c)]
            title = title_txt_dict[str(c)]
            if phase == 'train':
                pairs.append([query.strip(), claim.strip()])
                pairs.append([query.strip(), title.strip()])
                # pairs.append([query.strip(), title.strip()+" "+claim.strip()])
            else:
                pairs.append([query.strip(), claim.strip()])
                # pairs.append([query.strip(), title.strip()+" "+claim.strip()])
        
        cnt += 1

    return pairs


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

emb_type = 'distiluse-base-multilingual-cased'

emb_data = json.load(open(my_loc+'/bert_embs/'+emb_type+fname+'.json', 'r')) 
tr_embs = emb_data['train']['embs']
val_embs = emb_data['val']['embs']
cl_embs = emb_data['claims']['embs2']
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

## Cosine similarity 
cos_sim_tr = cosine_similarity(tr_embs, cl_embs)
cos_sim_val = cosine_similarity(val_embs, cl_embs)

tr_pos_pairs = get_pospair_feats(tr_pairs, tr_txt_dict, claim_txt_dict, title_txt_dict, 'train')
val_pos_pairs = get_pospair_feats(val_pairs, val_txt_dict, claim_txt_dict, title_txt_dict, 'val')

tr_neg_pairs = get_negpair_feats(tr_ids, tr_claim_dict, tr_txt_dict, claim_txt_dict, title_txt_dict, cos_sim_tr, args.neg_sample, 'train')
val_neg_pairs = get_negpair_feats(val_ids, val_claim_dict, val_txt_dict, claim_txt_dict, title_txt_dict, cos_sim_val, 1, 'val')


# extra_data = []
# with open('data/ibm_debater/evidence.txt','r', encoding='utf-8') as f:
#     for line in f:
#         _, claim, evid, _ = line.split('\t')
#         extra_data.append([claim, evid])

with open(my_loc+'/proc_data/train-pairs.csv', 'w', encoding='utf-8') as f:
    for prs in tr_pos_pairs:
        f.write('%s\t%s\t1\n'%(prs[0], prs[1]))

    # for prs in extra_data:
    #     f.write('%s\t%s\t1\n'%(prs[0], prs[1]))

    # for prs in tr_neg_pairs:
        # f.write('%s\t%s\t0\n'%(prs[0], prs[1]))


with open(my_loc+'/proc_data/val-pairs.csv', 'w', encoding='utf-8') as f:
    for prs in val_pos_pairs:
        f.write('%s\t%s\t1\n'%(prs[0], prs[1]))

    # for prs in val_neg_pairs:
        # f.write('%s\t%s\t0\n'%(prs[0], prs[1]))


num_epochs = 4
batch_size = 8
model_save_path = my_loc+'/models/finetune_claim_title_%s'%(fname)

sts_reader = STSBenchmarkDataReader(my_loc+'/proc_data/', s1_col_idx=0, s2_col_idx=1, score_col_idx=2, 
                                    normalize_scores=False, min_score=0, max_score=1)

model = SentenceTransformer(emb_type)
train_data = SentencesDataset(sts_reader.get_examples('train-pairs.csv'), model)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
train_loss = losses.MultipleNegativesRankingLoss(model=model)

dev_data = SentencesDataset(examples=sts_reader.get_examples('val-pairs.csv'), model=model)
dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=batch_size)
evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)

warmup_steps = math.ceil(len(train_data)*num_epochs/batch_size*0.1) #10% of train data for warm-up

model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=100,
          warmup_steps=warmup_steps,
          output_path=model_save_path)



model = SentenceTransformer(model_save_path)


