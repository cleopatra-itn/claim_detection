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

import json, re
import pandas as pd
import numpy as np

import spacy
import gensim
import gensim.downloader as api

import argparse

my_loc = os.path.dirname(__file__)

parser = argparse.ArgumentParser(description='Training for Word Embs + NEs')
parser.add_argument('--tag_type', type=int,
                    help='1,2,3,4')
parser.add_argument('--norm_type', type=int,
                    help='1,2,3,4')
parser.add_argument('--out_file', type=int, default=1,
                    help='1,2,3')


args = parser.parse_args()

tag_type = args.tag_type
norm_type = args.norm_type


def get_best_svm_model(feature_vector_train, label, feature_vector_valid):
    param_grid = [{'kernel':'linear', 'C': np.logspace(-3, 3, 20), 'gamma': [1]}, 
                  {'kernel':'rbf', 'C': np.logspace(-3, 3, 20), 
                  'gamma': np.logspace(-3, 3, 20)}]

    pca_list = [1.0,0.99,0.98,0.97,0.96,0.95]
    best_acc = 0.0
    best_model = 0
    best_prec = 0.0
    best_pca_nk = 0
    temp_xtrain = feature_vector_train
    temp_xval = feature_vector_valid
    for pca_nk in pca_list:
        if pca_nk != 1.0:
            pca = decomposition.PCA(n_components=pca_nk).fit(temp_xtrain)
            feature_vector_train = pca.transform(temp_xtrain)
            feature_vector_valid = pca.transform(temp_xval)
        for params in param_grid:
            for C in params['C']:
                for gamma in params['gamma']:
                    # Model with different parameters
                    model = svm.SVC(C=C, gamma=gamma, kernel=params['kernel'], random_state=42, class_weight='balanced')

                    # fit the training dataset on the classifier
                    model.fit(feature_vector_train, label)

                    # predict the labels on validation dataset
                    predictions = model.predict(feature_vector_valid)

                    acc = metrics.accuracy_score(predictions, val_y)

                    predicted_distance = model.decision_function(feature_vector_valid)
                    results_fpath = my_loc+'/results/temp9_%d.tsv'%(args.out_file)
                    with open(results_fpath, "w") as results_file:
                        for i, line in valDF.iterrows():
                            dist = predicted_distance[i]
                            results_file.write("{}\t{}\t{}\t{}\n".format('covid-19', line['tweet_id'],
                                                                         dist, "w2v_all"))

                    _, _, avg_precision, _, _ = evaluate('data/dev.tsv',results_fpath)

                    if round(avg_precision,4) >= round(best_prec,4) and round(acc,2) >= round(best_acc,2):
                        best_prec = avg_precision
                        best_acc = acc
                        best_model = model
                        best_pca_nk = pca_nk

    return best_acc, best_pca_nk, best_model


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

    return twit_x, np.array(twit_y), tweetDF


def get_ne_feat(tweet_list, ner_type):
    nes_feat = []
    for id in tweet_list:
        temp = np.zeros(len(ne_tags))
        proc_twit = tweet_list[id][ner_type]
        for wd in proc_twit:
            ne_tag = wd['label']
            if ne_tag in ne_tags:
                temp[ne_tags[ne_tag]] += 1
        
        # if sum(temp) > 0:
        #     temp = temp/sum(temp)

        nes_feat.append(temp)

    return np.array(nes_feat)


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
                rel = token.pos_+'-'+token.dep_+'-'+token.head.pos_
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
            rel = token.pos_+'-'+token.dep_+'-'+token.head.pos_

            if rel in dep_map:
                temp[dep_map[rel]] += 1

        if sum(temp) > 0:
            temp = temp/sum(temp)

        feats.append(temp)

    return np.array(feats)


nlp = spacy.load('en_core_web_lg')

train_data = json.load(open(my_loc+'/proc_data/train_data2.json', 'r'))
val_data = json.load(open(my_loc+'/proc_data/val_data2.json', 'r'))

if tag_type == 1:
    ne_tags = {'GPE':0, 'PERSON':1, 'ORG':2, 'NORP':3, 'LOC':4}
elif tag_type == 2:
    ne_tags = {'GPE':0, 'PERSON':1, 'ORG':2, 'NORP':3, 'LOC':4, 'DATE': 5}
elif tag_type == 3:
    ne_tags = {'GPE':0, 'PERSON':1, 'ORG':2, 'NORP':3, 'LOC':4,
            'CARDINAL':5, 'TIME':6, 'ORDINAL':7, 'FAC':8, 'MONEY': 9}
else:
    ne_tags = {'GPE':0, 'PERSON':1, 'ORG':2, 'NORP':3, 'LOC':4, 'DATE':5,
            'CARDINAL':6, 'TIME':7, 'ORDINAL':8, 'FAC':9, 'MONEY': 10}

pos_tags = {'NOUN':0, 'VERB':1, 'PROPN':2, 'ADJ':3, 'ADV':4, 'NUM':5,
            'ADP':6, 'PRON':7, 'DET':8, 'INTJ':9, 'AUX':10, 'PART':11,
            'CONJ':12, 'CCONJ':12, 'SCONJ':12}

if norm_type == 1:
    ner_type = 'ner_twit'
    pos_type = 'pos_twit'
    w2v_type = 'twit_clean'
elif norm_type == 2:
    ner_type = 'ner_twit_nostop'
    pos_type = 'pos_twit_nostop'
    w2v_type = 'twit_clean_nostop'
elif norm_type == 3:
    ner_type = 'ner_twit'
    pos_type = 'pos_twit'
    w2v_type = 'twit_clean_nostop'
else:
    ner_type = 'ner_twit_nostop'
    pos_type = 'pos_twit_nostop'
    w2v_type = 'twit_clean'

train_x, train_y, trainDF = get_tweet_data(train_data, w2v_type)
val_x, val_y, valDF = get_tweet_data(val_data, w2v_type)

train_nes = get_ne_feat(train_data, ner_type)
val_nes = get_ne_feat(val_data, ner_type)

train_pos = get_pos_feat(train_data, pos_type)
val_pos = get_pos_feat(val_data, pos_type)

dep_rels = get_dep_relations(w2v_type)
dep_map = {val:key for key,val in enumerate(list(dep_rels))}

train_dep = get_dep_feats(train_data, w2v_type)
val_dep = get_dep_feats(val_data, w2v_type)

train_ft_all = np.concatenate((train_nes, train_pos, train_dep), axis=1)
val_ft_all = np.concatenate((val_nes, val_pos, val_dep), axis=1)

all_res = []


wordvec_list = ['glove-twitter-25', 'glove-twitter-50', 'glove-twitter-100', 'glove-twitter-200']

for wordmod in wordvec_list:
    if wordmod == 'spacy':
        nlp = spacy.load('en_vectors_web_lg')
        ft_train = get_spacy_doc_vectors(train_x, nlp)
        ft_val = get_spacy_doc_vectors(val_x, nlp)
    else:
        wd_model = api.load(wordmod)
        sz = wordmod.split('-')[-1]
        ft_train = get_gensim_doc_vectors(train_x, wd_model, int(sz))
        ft_val = get_gensim_doc_vectors(val_x, wd_model, int(sz))


    ft_train = np.hstack((ft_train, train_ft_all))
    ft_val = np.hstack((ft_val, val_ft_all))

    accuracy, best_pca_nk, classifier = get_best_svm_model(ft_train, train_y, ft_val)

    if best_pca_nk != 1.0:
        pca = decomposition.PCA(n_components=best_pca_nk).fit(ft_train)
        ft_val = pca.transform(ft_val)

    print("SVM, %s+NE+PoS+DEP Accuracy: %.3f"%(wordmod, round(accuracy,3)))
    print("PCA No. Components: %.2f, Dim: %d"%(best_pca_nk, ft_val.shape[1]))
    print("C: %.3f, Gamma: %.3f, kernel: %s"%(classifier.C, classifier.gamma, classifier.kernel))

    predicted_distance = classifier.decision_function(ft_val)
    results_fpath = my_loc+'/results/task1_%s_all_svm_dev_%d.tsv'%(wordmod, args.out_file)
    with open(results_fpath, "w") as results_file:
        for i, line in valDF.iterrows():
            dist = predicted_distance[i]
            results_file.write("{}\t{}\t{}\t{}\n".format('covid-19', line['tweet_id'],
                dist, wordmod+'_all'))

    thresholds, precisions, avg_precision, reciprocal_rank, num_relevant = evaluate('data/dev.tsv', results_fpath)
    print("%s+NE+PoS+DEP SVM AVGP: %.4f\n"%(wordmod, round(avg_precision,4)))

    all_res.append([round(accuracy,3), round(avg_precision,4), best_pca_nk, ft_train.shape[1], ft_val.shape[1]])


with open(my_loc+'/file_results/w2v_all_%d.txt'%(args.out_file), 'a+') as f:
    for res in all_res:
        f.write("%.3f,%.4f,%.2f,%d,%d\n"%(res[0], res[1], res[2], res[3], res[4]))

    f.write('\n\n')





