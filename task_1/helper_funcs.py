from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

import sys, os

import json, re
import pandas as pd
import numpy as np

from nltk.corpus import stopwords

from transformers import BertTokenizer, BertModel
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


def get_text_processor(word_stats='twitter'):
    return TextPreProcessor(
            # terms that will be normalized , 'number','money', 'time','date' removed from below list
            normalize=['url', 'email', 'percent', 'phone', 'user'],
            # terms that will be annotated
            annotate={"hashtag", "allcaps", "elongated", "repeated",
                      'emphasis', 'censored'},
            fix_html=True,  # fix HTML tokens

            # corpus from which the word statistics are going to be used
            # for word segmentation
            segmenter=word_stats,

            # corpus from which the word statistics are going to be used
            # for spell correction
            corrector=word_stats,

            unpack_hashtags=True,  # perform word segmentation on hashtags
            unpack_contractions=True,  # Unpack contractions (can't -> can not)
            spell_correct_elong=False,  # spell correction for elongated words

            # select a tokenizer. You can use SocialTokenizer, or pass your own
            # the tokenizer, should take as input a string and return a list of tokens
            tokenizer=SocialTokenizer(lowercase=True).tokenize,

            # list of dictionaries, for replacing tokens extracted from the text,
            # with other expressions. You can pass more than one dictionaries.
            dicts=[emoticons]
        )




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
