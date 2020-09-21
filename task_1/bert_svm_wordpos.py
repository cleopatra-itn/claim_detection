## Features: Word Embeddings
## ## Models: SVM

import sys, os
sys.path.append('.')

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble, tree
from sklearn.preprocessing import StandardScaler

from scorer.main import evaluate
from nltk.corpus import stopwords

import json, pickle, time
import pandas as pd
import numpy as np

import spacy
import gensim
import gensim.downloader as api
from thundersvm import *

import argparse

my_loc = os.path.dirname(__file__)

parser = argparse.ArgumentParser(description='Training for Word Embs')
parser.add_argument('--normalize', type=int, default=1,
                    help='0,1')
parser.add_argument('--bert_type', type=int, default=0,
                    help='0,1,2,3')
parser.add_argument('--gpu_id', type=int, default=0,
                    help='0,1,2,3')

args = parser.parse_args()

def get_best_svm_model(feature_vector_train, label, feature_vector_valid, fname, emb_type):
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
                    results_fpath = my_loc+'/results/bert_word_pos_%s_%s_svm_norm%d.tsv'%(fname, emb_type, args.normalize)
                    with open(results_fpath, "w") as results_file:
                        for i, line in valDF.iterrows():
                            dist = predicted_distance[i][0]
                            results_file.write("{}\t{}\t{}\t{}\n".format('covid-19', line['tweet_id'],
                                                                         dist, "bert_wd_pos"))

                    _, _, avg_precision, _, _ = evaluate('data/dev.tsv',results_fpath)

                    if round(avg_precision,4) >= round(best_prec,4) and round(acc,2) >= round(best_acc,2):
                        best_prec = avg_precision
                        best_acc = acc
                        best_model = model
                        best_pca_nk = pca_nk

    return best_acc, best_pca_nk, best_model



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



nlp = spacy.load('en_core_web_lg')

train_dict = json.load(open(my_loc+'/proc_data/train_data.json', 'r', encoding='utf-8'))
val_dict = json.load(open(my_loc+'/proc_data/val_data.json', 'r', encoding='utf-8'))

train_y, trainDF = get_tweet_data(train_dict)
val_y, valDF = get_tweet_data(val_dict)


## Syntactic Features
pos_tags = {'NOUN':0, 'VERB':1, 'PROPN':2, 'ADJ':3, 'ADV':4, 'NUM':5}

pos_type = 'pos_wiki_nostop'
w2v_type = 'wiki_clean_nostop' 

train_pos = get_pos_feat(train_dict, pos_type)
val_pos = get_pos_feat(val_dict, pos_type)


## Bert Embeddings
# files = ['bert-base-uncased_raw_text', 'bert-base-uncased_proc_text', 
#           'bert-large-uncased_raw_text', 'bert-large-uncased_proc_text']
files = ['bert-large-uncased_raw_text']

fname = files[args.bert_type] 

data = json.load(open(my_loc+'/bert_embs/'+fname+'.json', 'r')) 
train_data = data['train']
val_data = data['val']
train_y = np.array(train_data['labels']).astype(np.int)
val_y = np.array(val_data['labels']).astype(np.int)

# emb_list = ['sent_word_catavg', 'sent_word_catavg_wostop', 'sent_word_sumavg',
#             'sent_word_sumavg_wostop', 'sent_emb_2_last', 'sent_emb_2_last_wostop',
#             'sent_emb_last', 'sent_emb_last_wostop']

emb_list = ['sent_word_catavg_wostop', 'sent_word_sumavg_wostop', 'sent_emb_2_last_wostop']

all_res = []

for emb_type in emb_list:
    since = time.time()
    ft_train = np.array(train_data[emb_type])
    ft_val = np.array(val_data[emb_type])

    if args.normalize:
        tr_norm = np.linalg.norm(ft_train, axis=1)
        tr_norm[tr_norm==0] = 1.0
        val_norm = np.linalg.norm(ft_val, axis=1)
        val_norm[val_norm==0] = 1.0
        ft_train = ft_train/tr_norm[:, np.newaxis]
        ft_val = ft_val/val_norm[:, np.newaxis]

    ft_train = np.concatenate((ft_train, train_pos), axis=1)
    ft_val = np.concatenate((ft_val, val_pos), axis=1)

    print(ft_train.shape, ft_val.shape)

    accuracy, best_pca_nk, classifier = get_best_svm_model(ft_train, train_y, ft_val, fname, emb_type)

    if best_pca_nk != 1.0:
        pca = decomposition.PCA(n_components=best_pca_nk).fit(ft_train)
        ft_val = pca.transform(ft_val)


    print("SVM, %s, %s Accuracy: %.3f"%(fname, emb_type, round(accuracy,3)))
    print("PCA No. Components: %.2f, Dim: %d"%(best_pca_nk, ft_val.shape[1]))
    print("C: %.3f, Gamma: %.3f, kernel: %s"%(classifier.C, classifier.gamma, classifier.kernel))

    predicted_distance = classifier.decision_function(ft_val)
    results_fpath = my_loc+'/results/bert_word_pos_%s_%s_svm_norm%d.tsv'%(fname, emb_type, args.normalize)
    with open(results_fpath, "w") as results_file:
        for i, line in valDF.iterrows():
            dist = predicted_distance[i][0]
            results_file.write("{}\t{}\t{}\t{}\n".format('covid-19', line['tweet_id'],
                dist, 'bert_wd_pos'))

    _, _, avg_precision, _, _ = evaluate('data/dev.tsv',results_fpath)
    print("%s, %s SVM AVGP: %.4f\n"%(fname, emb_type, round(avg_precision,4)))

    pickle.dump({'best_pca': best_pca_nk}, open(my_loc+'/models/'+fname+'_'+emb_type+'_pos_norm%s.pkl'%(args.normalize), 'wb'))
    classifier.save_to_file(my_loc+'/models/'+fname+'_'+emb_type+'_pos_norm%s.dt'%(args.normalize))

    all_res.append([emb_type,round(accuracy,3), round(avg_precision,4),
                    best_pca_nk, ft_train.shape[1], ft_val.shape[1]])
    
    print("Completed in: {} minutes\n".format((time.time()-since)/60.0))


with open(my_loc+'/file_results/bert_svm_word_pos_%s_norm%d.txt'%(fname, args.normalize), 'w') as f:
    for res in all_res:
        f.write("%s\t%.3f\t%.4f\t%.2f\t%d\t%d\n"%(res[0], res[1], res[2], res[3], res[4], res[5]))

    f.write('\n\n')