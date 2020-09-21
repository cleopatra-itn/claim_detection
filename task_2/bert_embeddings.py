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
    # Encode Text
    input_ids = torch.tensor([tokenizer.encode(tweet, add_special_tokens=True)]).to(device)

    tokenized_text = [tokenizer.decode(id) for id in input_ids.squeeze(0).cpu().numpy().tolist()]

    # Predict hidden states features for each layer
    with torch.no_grad():
        _, pooled_out, encoded_layers = model(input_ids)


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
        sent_emb_2_last, sent_emb_2_last_wostop, sent_emb_last, sent_emb_last_wostop, pooled_out.squeeze(0).cpu().numpy()


train_data = json.load(open(my_loc+'/proc_data/train.json', 'r'))
val_data = json.load(open(my_loc+'/proc_data/val.json', 'r'))
test_data = json.load(open(my_loc+'/proc_data/test.json', 'r'))
claim_data = json.load(open(my_loc+'/proc_data/claim_dict.json', 'r'))


data_type = {'train': train_data, 'val': val_data, 'test': test_data, 'claims': claim_data}

# bert_list = ['bert-base-uncased', 'bert-large-uncased']
# for bert_type in bert_list:
#     tokenizer = BertTokenizer.from_pretrained(bert_type)
#     model = BertModel.from_pretrained(bert_type, output_hidden_states=True)
#     model.to(device).eval()

#     embed_dict = {'train':{'sent_word_catavg':[], 'sent_word_catavg_wostop':[], 'sent_word_sumavg': [],
#                             'sent_word_sumavg_wostop': [], 'sent_emb_2_last': [], 'sent_emb_2_last_wostop': [],
#                             'sent_emb_last': [], 'sent_emb_last_wostop': [], 'pooled_out':[], 'id': [] }, 
#                     'val': {'sent_word_catavg':[], 'sent_word_catavg_wostop':[], 'sent_word_sumavg': [],
#                             'sent_word_sumavg_wostop': [], 'sent_emb_2_last': [], 'sent_emb_2_last_wostop': [],
#                             'sent_emb_last': [], 'sent_emb_last_wostop': [], 'pooled_out':[], 'id': [] },
#                     'test': {'sent_word_catavg':[], 'sent_word_catavg_wostop':[], 'sent_word_sumavg': [],
#                             'sent_word_sumavg_wostop': [], 'sent_emb_2_last': [], 'sent_emb_2_last_wostop': [],
#                             'sent_emb_last': [], 'sent_emb_last_wostop': [], 'pooled_out':[], 'id': [] },
#                     'claims': {'sent_word_catavg':[], 'sent_word_catavg_wostop':[], 'sent_word_sumavg': [],
#                             'sent_word_sumavg_wostop': [], 'sent_emb_2_last': [], 'sent_emb_2_last_wostop': [],
#                             'sent_emb_last': [], 'sent_emb_last_wostop': [], 'sent_word_catavg2':[], 'sent_word_catavg_wostop2':[], 'sent_word_sumavg2': [],
#                             'sent_word_sumavg_wostop2': [], 'sent_emb_2_last2': [], 'sent_emb_2_last_wostop2': [],
#                             'sent_emb_last2': [], 'sent_emb_last_wostop2': [], 'pooled_out': [], 'pooled_out2': [], 'id': []}
#                 }


#     for phase in data_type:
#         data = data_type[phase]
#         if phase == 'claims':
#             for id in data: 
#                 claim = data[id]['claim']
#                 title = data[id]['title']

#                 sent_word_catavg1, sent_word_catavg_wostop1, sent_word_sumavg1, sent_word_sumavg_wostop1, \
#                     sent_emb_2_last1, sent_emb_2_last_wostop1, sent_emb_last1, sent_emb_last_wostop1, pooled_out1 = get_word_sent_embedding(claim, model, tokenizer)

#                 sent_word_catavg2, sent_word_catavg_wostop2, sent_word_sumavg2, sent_word_sumavg_wostop2, \
#                     sent_emb_2_last2, sent_emb_2_last_wostop2, sent_emb_last2, sent_emb_last_wostop2, pooled_out2 = get_word_sent_embedding(title, model, tokenizer)


#                 embed_dict[phase]['sent_word_catavg'].append(sent_word_catavg1.tolist())
#                 embed_dict[phase]['sent_word_sumavg'].append(sent_word_sumavg1.tolist())
#                 embed_dict[phase]['sent_emb_2_last'].append(sent_emb_2_last1.tolist())
#                 embed_dict[phase]['sent_emb_last'].append(sent_emb_last1.tolist())
#                 embed_dict[phase]['sent_word_catavg_wostop'].append(sent_word_catavg_wostop1.tolist())
#                 embed_dict[phase]['sent_word_sumavg_wostop'].append(sent_word_sumavg_wostop1.tolist())
#                 embed_dict[phase]['sent_emb_2_last_wostop'].append(sent_emb_2_last_wostop1.tolist())
#                 embed_dict[phase]['sent_emb_last_wostop'].append(sent_emb_last_wostop1.tolist())
#                 embed_dict[phase]['pooled_out'].append(pooled_out1.tolist())
#                 embed_dict[phase]['sent_word_catavg2'].append(((sent_word_catavg1+sent_word_catavg2)/2).tolist())
#                 embed_dict[phase]['sent_word_sumavg2'].append(((sent_word_sumavg1+sent_word_sumavg2)/2).tolist())
#                 embed_dict[phase]['sent_emb_2_last2'].append(((sent_emb_2_last1+sent_emb_2_last2)/2).tolist())
#                 embed_dict[phase]['sent_emb_last2'].append(((sent_emb_last1+sent_emb_last2)/2).tolist())
#                 embed_dict[phase]['sent_word_catavg_wostop2'].append(((sent_word_catavg_wostop1+sent_word_catavg_wostop2)/2).tolist())
#                 embed_dict[phase]['sent_word_sumavg_wostop2'].append(((sent_word_sumavg_wostop1+sent_word_sumavg_wostop2)/2).tolist())
#                 embed_dict[phase]['sent_emb_2_last_wostop2'].append(((sent_emb_2_last_wostop1+sent_emb_2_last_wostop2)/2).tolist())
#                 embed_dict[phase]['sent_emb_last_wostop2'].append(((sent_emb_last_wostop1+sent_emb_last_wostop2)/2).tolist())
#                 embed_dict[phase]['pooled_out2'].append(((pooled_out1+pooled_out2)/2).tolist())
#                 embed_dict[phase]['id'].append(id)
#         else:
#             for id in data: 
#                 text = data[id]['text']

#                 sent_word_catavg, sent_word_catavg_wostop, sent_word_sumavg, sent_word_sumavg_wostop, \
#                     sent_emb_2_last, sent_emb_2_last_wostop, sent_emb_last, sent_emb_last_wostop, pooled_out = get_word_sent_embedding(text, model, tokenizer)

#                 embed_dict[phase]['sent_word_catavg'].append(sent_word_catavg.tolist())
#                 embed_dict[phase]['sent_word_sumavg'].append(sent_word_sumavg.tolist())
#                 embed_dict[phase]['sent_emb_2_last'].append(sent_emb_2_last.tolist())
#                 embed_dict[phase]['sent_emb_last'].append(sent_emb_last.tolist())
#                 embed_dict[phase]['sent_word_catavg_wostop'].append(sent_word_catavg_wostop.tolist())
#                 embed_dict[phase]['sent_word_sumavg_wostop'].append(sent_word_sumavg_wostop.tolist())
#                 embed_dict[phase]['sent_emb_2_last_wostop'].append(sent_emb_2_last_wostop.tolist())
#                 embed_dict[phase]['sent_emb_last_wostop'].append(sent_emb_last_wostop.tolist())
#                 embed_dict[phase]['pooled_out'].append(pooled_out.tolist())
#                 embed_dict[phase]['id'].append(id)


#     json.dump(embed_dict, open(my_loc+'/bert_embs/%s_raw_text.json'%(bert_type), 'w'))

#     ## Bert Embeddings on processed text 
#     embed_dict = {'train':{'sent_word_catavg':[], 'sent_word_catavg_wostop':[], 'sent_word_sumavg': [],
#                             'sent_word_sumavg_wostop': [], 'sent_emb_2_last': [], 'sent_emb_2_last_wostop': [],
#                             'sent_emb_last': [], 'sent_emb_last_wostop': [], 'pooled_out':[], 'id': [] }, 
#                     'val': {'sent_word_catavg':[], 'sent_word_catavg_wostop':[], 'sent_word_sumavg': [],
#                             'sent_word_sumavg_wostop': [], 'sent_emb_2_last': [], 'sent_emb_2_last_wostop': [],
#                             'sent_emb_last': [], 'sent_emb_last_wostop': [], 'pooled_out':[], 'id': [] },
#                     'test': {'sent_word_catavg':[], 'sent_word_catavg_wostop':[], 'sent_word_sumavg': [],
#                             'sent_word_sumavg_wostop': [], 'sent_emb_2_last': [], 'sent_emb_2_last_wostop': [],
#                             'sent_emb_last': [], 'sent_emb_last_wostop': [], 'pooled_out':[], 'id': [] },
#                     'claims': {'sent_word_catavg':[], 'sent_word_catavg_wostop':[], 'sent_word_sumavg': [],
#                             'sent_word_sumavg_wostop': [], 'sent_emb_2_last': [], 'sent_emb_2_last_wostop': [],
#                             'sent_emb_last': [], 'sent_emb_last_wostop': [], 'sent_word_catavg2':[], 'sent_word_catavg_wostop2':[], 'sent_word_sumavg2': [],
#                             'sent_word_sumavg_wostop2': [], 'sent_emb_2_last2': [], 'sent_emb_2_last_wostop2': [],
#                             'sent_emb_last2': [], 'sent_emb_last_wostop2': [], 'pooled_out': [], 'pooled_out2': [], 'id': []}
#                 }

#     for phase in data_type:
#         data = data_type[phase]    
#         if phase == 'claims':
#             for id in data:
#                 proc_claim = data[id]['claim_proc']
#                 proc_title = data[id]['title_proc']

#                 proc_claim = [word for word in proc_claim if not re.search(r'<(/?)[a-z]+>', word)]
#                 proc_title = [word for word in proc_title if not re.search(r'<(/?)[a-z]+>', word)]

#                 claim_text = ""
#                 for word in proc_claim:
#                     claim_text += word if word in [',', '.'] else " "+word

#                 title_text = ""
#                 for word in proc_title:
#                     title_text += word if word in [',', '.'] else " "+word

#                 sent_word_catavg1, sent_word_catavg_wostop1, sent_word_sumavg1, sent_word_sumavg_wostop1, \
#                     sent_emb_2_last1, sent_emb_2_last_wostop1, sent_emb_last1, sent_emb_last_wostop1, pooled_out1 = get_word_sent_embedding(claim_text, model, tokenizer)

#                 sent_word_catavg2, sent_word_catavg_wostop2, sent_word_sumavg2, sent_word_sumavg_wostop2, \
#                     sent_emb_2_last2, sent_emb_2_last_wostop2, sent_emb_last2, sent_emb_last_wostop2, pooled_out2 = get_word_sent_embedding(title_text, model, tokenizer)


#                 embed_dict[phase]['sent_word_catavg'].append(sent_word_catavg1.tolist())
#                 embed_dict[phase]['sent_word_sumavg'].append(sent_word_sumavg1.tolist())
#                 embed_dict[phase]['sent_emb_2_last'].append(sent_emb_2_last1.tolist())
#                 embed_dict[phase]['sent_emb_last'].append(sent_emb_last1.tolist())
#                 embed_dict[phase]['sent_word_catavg_wostop'].append(sent_word_catavg_wostop1.tolist())
#                 embed_dict[phase]['sent_word_sumavg_wostop'].append(sent_word_sumavg_wostop1.tolist())
#                 embed_dict[phase]['sent_emb_2_last_wostop'].append(sent_emb_2_last_wostop1.tolist())
#                 embed_dict[phase]['sent_emb_last_wostop'].append(sent_emb_last_wostop1.tolist())
#                 embed_dict[phase]['pooled_out'].append(pooled_out1.tolist())
#                 embed_dict[phase]['sent_word_catavg2'].append(((sent_word_catavg1+sent_word_catavg2)/2).tolist())
#                 embed_dict[phase]['sent_word_sumavg2'].append(((sent_word_sumavg1+sent_word_sumavg2)/2).tolist())
#                 embed_dict[phase]['sent_emb_2_last2'].append(((sent_emb_2_last1+sent_emb_2_last2)/2).tolist())
#                 embed_dict[phase]['sent_emb_last2'].append(((sent_emb_last1+sent_emb_last2)/2).tolist())
#                 embed_dict[phase]['sent_word_catavg_wostop2'].append(((sent_word_catavg_wostop1+sent_word_catavg_wostop2)/2).tolist())
#                 embed_dict[phase]['sent_word_sumavg_wostop2'].append(((sent_word_sumavg_wostop1+sent_word_sumavg_wostop2)/2).tolist())
#                 embed_dict[phase]['sent_emb_2_last_wostop2'].append(((sent_emb_2_last_wostop1+sent_emb_2_last_wostop2)/2).tolist())
#                 embed_dict[phase]['sent_emb_last_wostop2'].append(((sent_emb_last_wostop1+sent_emb_last_wostop2)/2).tolist())
#                 embed_dict[phase]['pooled_out2'].append(((pooled_out1+pooled_out2)/2).tolist())
#                 embed_dict[phase]['id'].append(id)

#         else:
#             for id in data:
#                 proc_text = data[id]['wiki_proc']
            
#                 text = ""
#                 for word in proc_text:
#                     text += word if word in [',', '.'] else " "+word


#                 sent_word_catavg, sent_word_catavg_wostop, sent_word_sumavg, sent_word_sumavg_wostop, \
#                     sent_emb_2_last, sent_emb_2_last_wostop, sent_emb_last, sent_emb_last_wostop, pooled_out = get_word_sent_embedding(text, model, tokenizer)

#                 embed_dict[phase]['sent_word_catavg'].append(sent_word_catavg.tolist())
#                 embed_dict[phase]['sent_word_sumavg'].append(sent_word_sumavg.tolist())
#                 embed_dict[phase]['sent_emb_2_last'].append(sent_emb_2_last.tolist())
#                 embed_dict[phase]['sent_emb_last'].append(sent_emb_last.tolist())
#                 embed_dict[phase]['sent_word_catavg_wostop'].append(sent_word_catavg_wostop.tolist())
#                 embed_dict[phase]['sent_word_sumavg_wostop'].append(sent_word_sumavg_wostop.tolist())
#                 embed_dict[phase]['sent_emb_2_last_wostop'].append(sent_emb_2_last_wostop.tolist())
#                 embed_dict[phase]['sent_emb_last_wostop'].append(sent_emb_last_wostop.tolist())
#                 embed_dict[phase]['pooled_out'].append(pooled_out.tolist())
#                 embed_dict[phase]['id'].append(id)

#     json.dump(embed_dict, open(my_loc+'/bert_embs/%s_proc_text.json'%(bert_type), 'w'))


## Bert Sentence embeddings from a sentence transformer

model_types = ['bert-base-nli-mean-tokens', 'bert-base-nli-cls-token', 'bert-base-nli-max-tokens',
                'bert-large-nli-mean-tokens', 'bert-large-nli-cls-token', 'bert-large-nli-max-tokens',
                'roberta-base-nli-mean-tokens', 'roberta-large-nli-mean-tokens', 'distilbert-base-nli-mean-tokens',
                'bert-base-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens', 'roberta-base-nli-stsb-mean-tokens', 
                'roberta-large-nli-stsb-mean-tokens', 'distilbert-base-nli-stsb-mean-tokens', 'distiluse-base-multilingual-cased',
                 'xlm-r-base-en-ko-nli-ststb', 'xlm-r-large-en-ko-nli-ststb']

for md in model_types:
    model = SentenceTransformer(md).to(device)

    embed_dict = {'train':{'id': [], 'embs': [] }, 
                    'val': {'id': [], 'embs': []},
                    'test': {'id': [], 'embs': []},
                    'claims': {'id': [], 'embs': [], 'embs2': []}}

    for phase in data_type:
        corpus = []
        corpus2 = []
        data = data_type[phase]
        for id in data:
            if phase == 'claims':
                claim = data[id]['claim']
                title = data[id]['title']

                corpus.append(claim)
                corpus2.append(title)

            else:
                text = data[id]['text']
                corpus.append(text)
            
            embed_dict[phase]['id'].append(id)
        
        if phase == 'claims':
            embeddings1 = model.encode(corpus, batch_size=32)
            embeddings2 = model.encode(corpus2, batch_size=32)
            for emb1, emb2 in zip(embeddings1, embeddings2):
                embed_dict[phase]['embs'].append(emb1.tolist())
                embed_dict[phase]['embs2'].append(((emb1+emb2)/2).tolist())
        else:
            embeddings = model.encode(corpus, batch_size=32)
            for emb in embeddings:
                embed_dict[phase]['embs'].append(emb.tolist())
            

    json.dump(embed_dict, open(my_loc+'/bert_embs/%s_raw_text.json'%(md), 'w'))


for md in model_types:
    model = SentenceTransformer(md).to(device)

    embed_dict = {'train':{'id': [], 'embs': [] }, 
                    'val': {'id': [], 'embs': []},
                    'test': {'id': [], 'embs': []},
                    'claims': {'id': [], 'embs': [], 'embs2': []}}

    for phase in data_type:
        corpus = []
        corpus2 = []
        data = data_type[phase]
        for id in data:
            if phase == 'claims':
                claim = data[id]['claim_proc']
                title = data[id]['title_proc']

                claim = [word for word in claim if not re.search(r'<(/?)[a-z]+>', word)]
                title = [word for word in title if not re.search(r'<(/?)[a-z]+>', word)]

                claim_text = ""
                for word in claim:
                    claim_text += word if word in [',', '.'] else " "+word

                title_text = ""
                for word in title:
                    title_text += word if word in [',', '.'] else " "+word

                corpus.append(claim_text)
                corpus2.append(title_text)

            else:
                proc_text = data[id]['wiki_proc']
                proc_text = [word for word in proc_text if not re.search(r'<(/?)[a-z]+>', word)]

                text = ""
                for word in proc_text:
                    text += word if word in [',', '.'] else " "+word

                corpus.append(text)
            
            embed_dict[phase]['id'].append(id)
            
        if phase == 'claims':
            embeddings1 = model.encode(corpus, batch_size=32)
            embeddings2 = model.encode(corpus2, batch_size=32)
            for emb1, emb2 in zip(embeddings1, embeddings2):
                embed_dict[phase]['embs'].append(emb1.tolist())
                embed_dict[phase]['embs2'].append(((emb1+emb2)/2).tolist())
        else:
            embeddings = model.encode(corpus, batch_size=32)
            for emb in embeddings:
                embed_dict[phase]['embs'].append(emb.tolist())

    json.dump(embed_dict, open(my_loc+'/bert_embs/%s_proc_text.json'%(md), 'w'))
    
