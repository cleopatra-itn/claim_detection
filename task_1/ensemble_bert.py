## Features: Word Embeddings + Feature Fusion
## ## Models: SVM

import sys, os
sys.path.append('.')

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

from scorer.main import evaluate
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import json, re, pickle, time
import pandas as pd
import numpy as np

import spacy
import gensim
import gensim.downloader as api
from thundersvm import *

import argparse

my_loc = os.path.dirname(__file__)

parser = argparse.ArgumentParser(description='Training for Word Embs + NEs')
parser.add_argument('--gpu_id', type=int, default=0,
                    help='0,1,2,3')

args = parser.parse_args()


def get_tweet_data(tweet_list):
    twit_y, twit_id = [], []
    for id in tweet_list:
        twit_id.append(id)
        twit_y.append(tweet_list[id]['worthy'])

    tweetDF = pd.DataFrame()
    tweetDF['label'] = twit_y
    tweetDF['tweet_id'] = twit_id

    return np.array(twit_y).astype(np.int), tweetDF


def get_pos_feat(tweet_list, pos_type):
    pos_feat = []
    for id in tweet_list:
        temp = np.zeros(len(pos_tags))
        proc_twit = tweet_list[id][pos_type]
        for wd in proc_twit:
            pos = wd.split('_')[1]
            if pos in pos_tags:
                temp[pos_tags[pos]] += 1

        if sum(temp) > 0:
            temp = temp/sum(temp)

        pos_feat.append(temp)

    return np.array(pos_feat)



def get_dep_relations(train_data, w2v_type):
    pos_tags = ['ADJ', 'ADV', 'NOUN', 'PROPN', 'VERB', 'NUM']
    dep_dict = set()
    for id in train_data:
        words = train_data[id][w2v_type]
        sent = " ".join(words)

        doc = nlp(sent)

        for token in doc:
            if token.pos_ in pos_tags and token.head.pos_ in pos_tags:
                # rel = token.pos_+'-'+token.dep_+'-'+token.head.pos_
                rel = token.pos_+'-'+token.dep_
                dep_dict.add(rel)

    return dep_dict


def get_dep_feats(tweet_list, w2v_type):
    feats = []
    for id in tweet_list:
        temp = np.zeros(len(dep_map))
        words = tweet_list[id][w2v_type]
        sent = " ".join(words)

        doc = nlp(sent)

        for token in doc:
            # rel = token.pos_+'-'+token.dep_+'-'+token.head.pos_
            rel = token.pos_+'-'+token.dep_
            if rel in dep_map:
                temp[dep_map[rel]] += 1


        if sum(temp) > 0:
            temp = temp/sum(temp)

        feats.append(temp)

    return np.array(feats)


nlp = spacy.load('en_core_web_lg')

train_dict = json.load(open(my_loc+'/proc_data/train_data.json', 'r', encoding='utf-8'))
val_dict = json.load(open(my_loc+'/proc_data/val_data.json', 'r', encoding='utf-8'))

train_y, trainDF = get_tweet_data(train_dict)
val_y, valDF = get_tweet_data(val_dict)

pos_tags = {'NOUN':0, 'VERB':1, 'PROPN':2, 'ADJ':3, 'ADV':4, 'NUM':5}

data = json.load(open(my_loc+'/bert_embs/bert-large-uncased_raw_text.json', 'r')) 
train_data = data['train']
val_data = data['val']
train_y = np.array(train_data['labels']).astype(np.int)
val_y = np.array(val_data['labels']).astype(np.int)

pos_type = 'pos_wiki'
w2v_type = 'wiki_clean'

train_pos = get_pos_feat(train_dict, pos_type)
val_pos = get_pos_feat(val_dict, pos_type)

dep_rels = get_dep_relations(train_dict, w2v_type)
dep_map = {val:key for key,val in enumerate(list(dep_rels))}

train_dep = get_dep_feats(train_dict, w2v_type)
val_dep = get_dep_feats(val_dict, w2v_type)


bert_list = ['sent_word_catavg_wostop', 'sent_word_sumavg',
            'sent_word_sumavg_wostop']
# bert_list = ['sent_word_catavg_wostop','sent_word_sumavg_wostop', 'sent_emb_2_last_wostop']

pred_all = []
desc_all = []

for emb_type in bert_list:
    
    ft_train = np.array(train_data[emb_type])
    ft_val = np.array(val_data[emb_type])

    tr_norm = np.linalg.norm(ft_train, axis=1)
    tr_norm[tr_norm==0] = 1.0
    val_norm = np.linalg.norm(ft_val, axis=1)
    val_norm[val_norm==0] = 1.0
    ft_train = ft_train/tr_norm[:, np.newaxis]
    ft_val = ft_val/val_norm[:, np.newaxis]

    ft_train = np.concatenate((ft_train, train_pos, train_dep), axis=1)
    ft_val = np.concatenate((ft_val, val_pos, val_dep), axis=1)

    model_params = pickle.load(open(my_loc+'/models/bert-large-uncased_raw_text_%s_posdep_norm1.pkl'%(emb_type),'rb'))
    best_pca = model_params['best_pca']
    print(best_pca)

    if best_pca != 1.0:
        pca = decomposition.PCA(n_components=best_pca).fit(ft_train)
        ft_val = pca.transform(ft_val)

    svm_model = SVC()
    svm_model.load_from_file(my_loc+'/models/bert-large-uncased_raw_text_%s_posdep_norm1.dt'%(emb_type))

    print("Model %s ACC: %.3f"%(emb_type, svm_model.score(ft_val, val_y)))

    pred_all.append(svm_model.predict(ft_val))
    desc_all.append(np.squeeze(svm_model.decision_function(ft_val),1))

pred_all = np.array(pred_all).astype(np.int)
desc_all = np.array(desc_all)


final_pred = np.ceil(np.mean(pred_all, axis=0)).astype(np.int)
# final_desc = np.mean(desc_all, axis=0)

final_desc = np.zeros(len(final_pred))
final_desc[final_pred==1] = np.max(desc_all, axis=0)[final_pred==1]
final_desc[final_pred==0] = np.min(desc_all, axis=0)[final_pred==0]


print("Ensemble ACC: %.3f"%(sum(val_y==final_pred)/len(val_y)))


results_fpath = my_loc+'/results/task1_ensemble_bert_posdep_svm_dev.tsv'
with open(results_fpath, "w") as results_file:
    for i, line in valDF.iterrows():
        dist = final_desc[i]
        results_file.write("{}\t{}\t{}\t{}\n".format('covid-19', line['tweet_id'],
                                                        dist, "bert_posdep"))

_, _, avg_precision, _, _ = evaluate('data/dev.tsv',results_fpath)

print("Ensemble Precision: %.4f"%(avg_precision))


