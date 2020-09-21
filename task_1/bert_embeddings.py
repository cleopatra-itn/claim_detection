## Features: Word Embeddings
## ## Models: SVM

import sys, os

import json, re
import pandas as pd
import numpy as np

from nltk.corpus import stopwords

from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

my_loc = os.path.dirname(__file__)

df_stopwords = set(stopwords.words('english'))

def avg_wordembs_wostop(tokens, word_embs, emb_sz):
    embs = []
    for i in range (len(tokens)):
        if tokens[i] not in df_stopwords:
            embs.append(word_embs[i])

    embs = np.array(embs)
    if not np.any(embs):
        return np.zeros(emb_sz).astype(np.float) 

    return np.mean(embs, axis=0)


def get_word_sent_embedding(tweet, model, tokenizer):
    # Split the sentence into tokens.
    tokenized_text = tokenizer.tokenize(tweet)

    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Mark each of the 22 tokens as belonging to sentence "1".
    segments_ids = [1] * len(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens]).to(device)
    segments_tensors = torch.tensor([segments_ids]).to(device)

    # Predict hidden states features for each layer
    with torch.no_grad():
        _, _, encoded_layers = model(tokens_tensor, segments_tensors)


    # Concatenate the tensors for all layers. We use `stack` here to
    # create a new dimension in the tensor.
    token_embeddings = torch.stack(encoded_layers, dim=0)

    # Remove dimension 1, the "batches".
    token_embeddings = torch.squeeze(token_embeddings, dim=1)

    # Swap dimensions 0 and 1.
    token_embeddings = token_embeddings.permute(1,0,2)

    # Stores the token vectors, with shape [22 x 3,072]
    token_vecs_cat = []

    # `token_embeddings` is a [22 x 12 x 768] tensor.
    # For each token in the sentence...
    for token in token_embeddings:
    
        # `token` is a [12 x 768] tensor

        # Concatenate the vectors (that is, append them together) from the last 
        # four layers.
        # Each layer vector is 768 values, so `cat_vec` is length 3,072.
        cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
        
        # Use `cat_vec` to represent `token`.
        token_vecs_cat.append(cat_vec.cpu().numpy())

    sent_word_catavg = np.mean(token_vecs_cat, axis=0)
    sent_word_catavg_wostop = avg_wordembs_wostop(tokenized_text,token_vecs_cat, len(token_vecs_cat[0]))

    # Stores the token vectors, with shape [22 x 768]
    token_vecs_sum = []

    # `token_embeddings` is a [22 x 12 x 768] tensor.

    # For each token in the sentence...
    for token in token_embeddings:

        # `token` is a [12 x 768] tensor

        # Sum the vectors from the last four layers.
        sum_vec = torch.sum(token[-4:], dim=0)
        
        # Use `sum_vec` to represent `token`.
        token_vecs_sum.append(sum_vec.cpu().numpy())

    sent_word_sumavg = np.mean(token_vecs_sum, axis=0)
    sent_word_sumavg_wostop = avg_wordembs_wostop(tokenized_text,token_vecs_sum, len(token_vecs_sum[0]))

    # `token_vecs` is a tensor with shape [22 x 768]
    token_vecs = encoded_layers[-2][0]

    # Calculate the average of all 22 token vectors.
    sent_emb_2_last = torch.mean(token_vecs, dim=0).cpu().numpy()
    sent_emb_2_last_wostop = avg_wordembs_wostop(tokenized_text,token_vecs.cpu().numpy(), len(token_vecs[0]))

    # `token_vecs` is a tensor with shape [22 x 768]
    token_vecs = encoded_layers[-1][0]

    # Calculate the average of all 22 token vectors.
    sent_emb_last = torch.mean(token_vecs, dim=0).cpu().numpy()
    sent_emb_last_wostop = avg_wordembs_wostop(tokenized_text,token_vecs.cpu().numpy(), len(token_vecs[0]))

    return sent_word_catavg, sent_word_catavg_wostop, sent_word_sumavg, sent_word_sumavg_wostop, \
        sent_emb_2_last, sent_emb_2_last_wostop, sent_emb_last, sent_emb_last_wostop


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
            proc_text = data[id]['wiki_proc']
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


## Bert Sentence embeddings from a sentence transformer

model_types = ['bert-base-nli-mean-tokens', 'bert-base-nli-cls-token', 'bert-base-nli-max-tokens',
                'bert-large-nli-mean-tokens', 'bert-large-nli-cls-token', 'bert-large-nli-max-tokens',
                'roberta-base-nli-mean-tokens', 'roberta-large-nli-mean-tokens', 'distilbert-base-nli-mean-tokens',
                'bert-base-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens', 'roberta-base-nli-stsb-mean-tokens', 
                'roberta-large-nli-stsb-mean-tokens' 'distilbert-base-nli-stsb-mean-tokens', 'distiluse-base-multilingual-cased']

for md in model_types:
    model = SentenceTransformer(md).to(device).eval()

    embed_dict = {'train':{'embs': [], 'labels': [] }, 
                    'val': {'embs': [], 'labels': [] }, 
                    'test': {'embs': []}}
    data_type = {'train': train_data, 'val': val_data, 'test': test_data}

    for phase in data_type:
        corpus = []
        data = data_type[phase]
        for id in data:
            text = data[id]['text']
            corpus.append(text)
            if phase != 'test':
                embed_dict[phase]['labels'].append(data[id]['worthy'])
        
        embeddings = model.encode(corpus, batch_size=32)
        for emb in embeddings:
            embed_dict[phase]['embs'].append(emb.tolist())

    json.dump(embed_dict, open(my_loc+'/bert_embs/%s_raw_text.json'%(md), 'w'))


for md in model_types:
    model = SentenceTransformer(md).to(device)

    embed_dict = {'train':{'embs': [], 'labels': [] }, 
                    'val': {'embs': [], 'labels': [] }, 
                    'test': {'embs': []}}
    data_type = {'train': train_data, 'val': val_data, 'test': test_data}

    for phase in data_type:
        corpus = []
        data = data_type[phase]
        for id in data:
            proc_text = data[id]['wiki_proc']
            proc_text = [word for word in proc_text if not re.search(r'<(/?)[a-z]+>', word)]
            text = ""
            for word in proc_text:
                text += word if word in [',', '.'] else " "+word

            corpus.append(text)
            if phase != 'test':
                embed_dict[phase]['labels'].append(data[id]['worthy'])
        
        embeddings = model.encode(corpus, batch_size=32)
        for emb in embeddings:
            embed_dict[phase]['embs'].append(emb.tolist())

    json.dump(embed_dict, open(my_loc+'/bert_embs/%s_proc_text.json'%(md), 'w'))
    