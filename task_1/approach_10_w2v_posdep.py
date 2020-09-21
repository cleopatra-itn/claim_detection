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
parser.add_argument('--tag_type', type=int, default=2,
                    help='1,2,3,4')
parser.add_argument('--norm_type', type=int, default=4,
                    help='1,2,3,4')
parser.add_argument('--gpu_id', type=int, default=0,
                    help='0,1,2,3')

args = parser.parse_args()

tag_type = args.tag_type
norm_type = args.norm_type


def get_best_svm_model(feature_vector_train, label, feature_vector_valid, wordmod):
    # param_grid = [{'kernel':'linear', 'C': np.logspace(-2, 2, 10), 'gamma': [1]}, 
    #               {'kernel':'rbf', 'C': np.logspace(-2, 2, 10), 
    #               'gamma': np.logspace(-2, 2, 10)}]
    param_grid = [{'kernel':'rbf', 'C': np.logspace(-3, 3, 30), 
                  'gamma': np.logspace(-3, 3, 30)}]

    pca_list = [1.0,0.99,0.98,0.97,0.96,0.95]
    best_acc = 0.0
    best_model = 0
    best_prec = 0.0
    best_pca_nk = 0
    temp_xtrain = feature_vector_train
    temp_xval = feature_vector_valid
    for pca_nk in pca_list:
        print(pca_nk)
        if pca_nk != 1.0:
            pca = decomposition.PCA(n_components=pca_nk).fit(temp_xtrain)
            feature_vector_train = pca.transform(temp_xtrain)
            feature_vector_valid = pca.transform(temp_xval)
        for params in param_grid:
            for C in params['C']:
                for gamma in params['gamma']:
                    # Model with different parameters
                    model = SVC(C=C, gamma=gamma, kernel=params['kernel'], random_state=42, class_weight='balanced', gpu_id=args.gpu_id)

                    # fit the training dataset on the classifier
                    model.fit(feature_vector_train, label)

                    # predict the acc on validation dataset
                    acc = model.score(feature_vector_valid, val_y)

                    predicted_distance = model.decision_function(feature_vector_valid)
                    results_fpath = my_loc+'/results/task1_%s_posdep_svm_dev_%d_%d.tsv'%(wordmod, args.tag_type, args.norm_type)
                    with open(results_fpath, "w") as results_file:
                        for i, line in valDF.iterrows():
                            dist = predicted_distance[i][0]
                            results_file.write("{}\t{}\t{}\t{}\n".format('covid-19', line['tweet_id'],
                                                                         dist, "w2v_posdep"))

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


nlp = spacy.load('en_core_web_lg')

train_data = json.load(open(my_loc+'/proc_data/train_data.json', 'r', encoding='utf-8'))
val_data = json.load(open(my_loc+'/proc_data/val_data.json', 'r', encoding='utf-8'))


if tag_type == 1:
    pos_tags = {'NOUN':0, 'VERB':1, 'PROPN':2, 'ADJ':3, 'ADV':4}
elif tag_type == 2:
    pos_tags = {'NOUN':0, 'VERB':1, 'PROPN':2, 'ADJ':3, 'ADV':4, 'NUM':5,
            'ADP':6, 'PRON':7}
elif tag_type == 3:
    pos_tags = {'NOUN':0, 'VERB':1, 'PROPN':2, 'ADJ':3, 'ADV':4, 'NUM':5,
            'ADP':6, 'PRON':7, 'DET':8, 'INTJ':9, 'AUX':10, 'PART':11}
else:
    pos_tags = {'NOUN':0, 'VERB':1, 'PROPN':2, 'ADJ':3, 'ADV':4, 'NUM':5,
            'ADP':6, 'PRON':7, 'DET':8, 'INTJ':9, 'AUX':10, 'PART':11,
            'CONJ':12, 'CCONJ':12, 'SCONJ':12}


if norm_type == 1:
    pos_type = 'pos_wiki'
    w2v_type = 'wiki_clean'
elif norm_type == 2:
    pos_type = 'pos_wiki_nostop'
    w2v_type = 'wiki_clean_nostop'
elif norm_type == 3:
    pos_type = 'pos_wiki'
    w2v_type = 'wiki_clean_nostop'
else:
    pos_type = 'pos_wiki_nostop'
    w2v_type = 'wiki_clean'

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

all_res = []


# wordvec_list = ['spacy','glove-twitter-25', 'glove-twitter-50', 'glove-twitter-100', 'glove-twitter-200',
#                 'glove-wiki-gigaword-50', 'glove-wiki-gigaword-100', 'glove-wiki-gigaword-200',
#                 'glove-wiki-gigaword-300', 'fasttext-wiki-news-subwords-300', 'word2vec-google-news-300']

wordvec_list = ['glove-twitter-25', 'glove-twitter-50', 'glove-twitter-100', 'glove-twitter-200']


for wordmod in wordvec_list:
    since = time.time()
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

    print(ft_train.shape, ft_val.shape)

    accuracy, best_pca_nk, classifier = get_best_svm_model(ft_train, train_y, ft_val, wordmod)

    if best_pca_nk != 1.0:
        pca = decomposition.PCA(n_components=best_pca_nk).fit(ft_train)
        ft_val = pca.transform(ft_val)

    print("SVM, %s+PoS+DEP Accuracy: %.3f"%(wordmod, round(accuracy,3)))
    print("PCA No. Components: %.2f, Dim: %d"%(best_pca_nk, ft_val.shape[1]))
    print("C: %.3f, Gamma: %.3f, kernel: %s"%(classifier.C, classifier.gamma, classifier.kernel))

    predicted_distance = classifier.decision_function(ft_val)
    results_fpath = my_loc+'/results/task1_%s_posdep_svm_dev_%d_%d.tsv'%(wordmod, args.tag_type, args.norm_type)
    with open(results_fpath, "w") as results_file:
        for i, line in valDF.iterrows():
            dist = predicted_distance[i][0]
            results_file.write("{}\t{}\t{}\t{}\n".format('covid-19', line['tweet_id'],
                dist, wordmod+'_posdep'))

    thresholds, precisions, avg_precision, reciprocal_rank, num_relevant = evaluate('data/dev.tsv', results_fpath)
    print("%s+PoS+DEP SVM AVGP: %.4f\n"%(wordmod, round(avg_precision,4)))

    pickle.dump({'best_pca': best_pca_nk, 'pos_type': pos_type, 'w2v_type': w2v_type, 'pos_tags': pos_tags}, 
                    open(my_loc+'/models/'+wordmod+'_posdep_%d_%d.pkl'%(args.tag_type, args.norm_type), 'wb'))
    classifier.save_to_file(my_loc+'/models/'+wordmod+'_posdep_%d_%d.dt'%(args.tag_type, args.norm_type))

    all_res.append([wordmod,round(accuracy,3), round(avg_precision,4), best_pca_nk, ft_train.shape[1], ft_val.shape[1]])

    print("Completed in: {} minutes\n".format((time.time()-since)/60.0))


with open(my_loc+'/file_results/w2v_posdep_%d_%d.txt'%(args.tag_type, args.norm_type), 'w') as f:
    for res in all_res:
        f.write("%s\t%.3f\t%.4f\t%.2f\t%d\t%d\n"%(res[0], res[1], res[2], res[3], res[4], res[5]))

    f.write('\n\n')

