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
    twit_x, twit_y, twit_id = [], [], []
    for id in tweet_list:
        twit_id.append(id)
        twit_y.append(tweet_list[id]['worthy'])
        twit_x.append(" ".join(tweet_list[id][w2v_type]))

    tweetDF = pd.DataFrame()
    tweetDF['text'] = twit_x
    tweetDF['label'] = twit_y
    tweetDF['tweet_id'] = twit_id

    return twit_x, np.array(twit_y).astype(np.int), tweetDF


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


def get_dep_relations(w2v_type):
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



train_data = json.load(open(my_loc+'/proc_data/train_data.json', 'r', encoding='utf-8'))
val_data = json.load(open(my_loc+'/proc_data/val_data.json', 'r', encoding='utf-8'))

nlp = spacy.load('en_core_web_lg')

pos_type = 'pos_twit_nostop'
w2v_type = 'twit_clean'

pos_tags = {'NOUN':0, 'VERB':1, 'PROPN':2, 'ADJ':3, 'ADV':4, 'NUM':5,
            'ADP':6, 'PRON':7}

train_x, train_y, trainDF = get_tweet_data(train_data, w2v_type)
val_x, val_y, valDF = get_tweet_data(val_data, w2v_type)

train_pos = get_pos_feat(train_data, pos_type)
val_pos = get_pos_feat(val_data, pos_type)

dep_rels = get_dep_relations(w2v_type)
dep_map = {val:key for key,val in enumerate(list(dep_rels))}

train_dep = get_dep_feats(train_data, w2v_type)
val_dep = get_dep_feats(val_data, w2v_type)

train_ft_all = np.concatenate((train_pos, train_dep), axis=1)
val_ft_all = np.concatenate((val_pos, val_dep), axis=1)


wordvec_list = ['glove-twitter-25', 'glove-twitter-50', 'glove-twitter-100']

pred_all = []
desc_all = []

for wordmod in wordvec_list:
    since = time.time()
    wd_model = api.load(wordmod)
    sz = wordmod.split('-')[-1]
    ft_train = get_gensim_doc_vectors(train_x, wd_model, int(sz))
    ft_val = get_gensim_doc_vectors(val_x, wd_model, int(sz))

    ft_train = np.hstack((ft_train, train_ft_all))
    ft_val = np.hstack((ft_val, val_ft_all))

    model_params = pickle.load(open(my_loc+'/models/%s_posdep_2_4.pkl'%(wordmod),'rb'))

    best_pca = model_params['best_pca']

    if best_pca != 1.0:
        pca = decomposition.PCA(n_components=best_pca).fit(ft_train)
        ft_val = pca.transform(ft_val)

    svm_model = SVC()
    svm_model.load_from_file(my_loc+'/models/%s_posdep_2_4.dt'%(wordmod))

    print("Model %s ACC: %.3f"%(wordmod, svm_model.score(ft_val, val_y)))

    pred_all.append(svm_model.predict(ft_val))
    desc_all.append(np.squeeze(svm_model.decision_function(ft_val),1))


pred_all = np.array(pred_all).astype(np.int)
desc_all = np.array(desc_all)

final_pred = np.ceil(np.mean(pred_all, axis=0)).astype(np.int)
final_desc = np.mean(desc_all, axis=0)

print("Ensemble ACC: %.3f"%(sum(val_y==final_pred)/len(val_y)))


results_fpath = my_loc+'/results/task1_ensemble_posdep_svm_dev_%d_%d.tsv'%(2, 4)
with open(results_fpath, "w") as results_file:
    for i, line in valDF.iterrows():
        dist = final_desc[i]
        results_file.write("{}\t{}\t{}\t{}\n".format('covid-19', line['tweet_id'],
                                                        dist, "w2v_posdep"))

_, _, avg_precision, _, _ = evaluate('data/dev.tsv',results_fpath)

print("Ensemble Precision: %.3f"%(avg_precision))


