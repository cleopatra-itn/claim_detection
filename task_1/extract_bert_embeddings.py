## Features: BERT Embeddings
## ## Models: SVM

import sys, os

import json, re
import pandas as pd
import numpy as np

from nltk.corpus import stopwords

from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
import torch

from helper_funcs import get_word_sent_embedding

my_loc = os.path.dirname(__file__)

train_data = json.load(open(my_loc+'/proc_data/train_data.json', 'r', encoding='utf-8'))
val_data = json.load(open(my_loc+'/proc_data/val_data.json', 'r', encoding='utf-8'))
test_data = json.load(open(my_loc+'/proc_data/test_data.json', 'r', encoding='utf-8'))

bert_list = ['bert-base-uncased', 'bert-large-uncased']
for bert_type in bert_list:
    tokenizer = BertTokenizer.from_pretrained(bert_type)
    model = BertModel.from_pretrained(bert_type, output_hidden_states=True)
    model.to(device).eval()

    embed_dict = {'train':{'sent_word_catavg':[], 'sent_word_catavg_wostop':[], 'sent_word_sumavg': [],
                            'sent_word_sumavg_wostop': [], 'sent_emb_2_last': [], 'sent_emb_2_last_wostop': [],
                            'sent_emb_last': [], 'sent_emb_last_wostop': [], 'labels': [] }, 
                    'val': {'sent_word_catavg':[], 'sent_word_catavg_wostop':[], 'sent_word_sumavg': [],
                            'sent_word_sumavg_wostop': [], 'sent_emb_2_last': [], 'sent_emb_2_last_wostop': [],
                            'sent_emb_last': [], 'sent_emb_last_wostop': [], 'labels': [] },
                    'test': {'sent_word_catavg':[], 'sent_word_catavg_wostop':[], 'sent_word_sumavg': [],
                            'sent_word_sumavg_wostop': [], 'sent_emb_2_last': [], 'sent_emb_2_last_wostop': [],
                            'sent_emb_last': [], 'sent_emb_last_wostop': []}
                }

    data_type = {'train': train_data, 'val': val_data, 'test': test_data}

    for phase in data_type:
        data = data_type[phase]
        for id in data:
            text = data[id]['text']
            marked_text = "[CLS] "+text+" [SEP]"

            sent_word_catavg, sent_word_catavg_wostop, sent_word_sumavg, sent_word_sumavg_wostop, \
            sent_emb_2_last, sent_emb_2_last_wostop, sent_emb_last, sent_emb_last_wostop = get_word_sent_embedding(marked_text, model, tokenizer)

            embed_dict[phase]['sent_word_catavg'].append(sent_word_catavg.tolist())
            embed_dict[phase]['sent_word_sumavg'].append(sent_word_sumavg.tolist())
            embed_dict[phase]['sent_emb_2_last'].append(sent_emb_2_last.tolist())
            embed_dict[phase]['sent_emb_last'].append(sent_emb_last.tolist())
            embed_dict[phase]['sent_word_catavg_wostop'].append(sent_word_catavg_wostop.tolist())
            embed_dict[phase]['sent_word_sumavg_wostop'].append(sent_word_sumavg_wostop.tolist())
            embed_dict[phase]['sent_emb_2_last_wostop'].append(sent_emb_2_last_wostop.tolist())
            embed_dict[phase]['sent_emb_last_wostop'].append(sent_emb_last_wostop.tolist())
            if phase != 'test':
                embed_dict[phase]['labels'].append(data[id]['worthy'])


    json.dump(embed_dict, open(my_loc+'/bert_embs/%s_raw_text.json'%(bert_type), 'w'))

    ## Bert Embeddings on processed text 
    embed_dict = {'train':{'sent_word_catavg':[], 'sent_word_catavg_wostop':[], 'sent_word_sumavg': [],
                            'sent_word_sumavg_wostop': [], 'sent_emb_2_last': [], 'sent_emb_2_last_wostop': [],
                            'sent_emb_last': [], 'sent_emb_last_wostop': [], 'labels': [] }, 
                    'val': {'sent_word_catavg':[], 'sent_word_catavg_wostop':[], 'sent_word_sumavg': [],
                            'sent_word_sumavg_wostop': [], 'sent_emb_2_last': [], 'sent_emb_2_last_wostop': [],
                            'sent_emb_last': [], 'sent_emb_last_wostop': [], 'labels': [] },
                    'test': {'sent_word_catavg':[], 'sent_word_catavg_wostop':[], 'sent_word_sumavg': [],
                            'sent_word_sumavg_wostop': [], 'sent_emb_2_last': [], 'sent_emb_2_last_wostop': [],
                            'sent_emb_last': [], 'sent_emb_last_wostop': []}
                }

    for phase in data_type:
        data = data_type[phase]
        for id in data:
            proc_text = data[id]['twit_proc']
            proc_text = [word for word in proc_text if not re.search(r'<(/?)[a-z]+>', word)]
            text = ""
            for word in proc_text:
                text += word if word in [',', '.'] else " "+word

            marked_text = "[CLS] "+text+" [SEP]"

            sent_word_catavg, sent_word_catavg_wostop, sent_word_sumavg, sent_word_sumavg_wostop, \
            sent_emb_2_last, sent_emb_2_last_wostop, sent_emb_last, sent_emb_last_wostop = get_word_sent_embedding(marked_text, model, tokenizer)

            embed_dict[phase]['sent_word_catavg'].append(sent_word_catavg.tolist())
            embed_dict[phase]['sent_word_sumavg'].append(sent_word_sumavg.tolist())
            embed_dict[phase]['sent_emb_2_last'].append(sent_emb_2_last.tolist())
            embed_dict[phase]['sent_emb_last'].append(sent_emb_last.tolist())
            embed_dict[phase]['sent_word_catavg_wostop'].append(sent_word_catavg_wostop.tolist())
            embed_dict[phase]['sent_word_sumavg_wostop'].append(sent_word_sumavg_wostop.tolist())
            embed_dict[phase]['sent_emb_2_last_wostop'].append(sent_emb_2_last_wostop.tolist())
            embed_dict[phase]['sent_emb_last_wostop'].append(sent_emb_last_wostop.tolist())
            if phase != 'test':
                embed_dict[phase]['labels'].append(data[id]['worthy'])

    json.dump(embed_dict, open(my_loc+'/bert_embs/%s_proc_text.json'%(bert_type), 'w'))