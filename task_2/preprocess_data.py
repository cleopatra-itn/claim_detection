import sys
sys.path.append('.')

from helper_funcs import *
import spacy
import re, os
from nltk.corpus import stopwords
import json
from urllib import request
from bs4 import BeautifulSoup as bs
from urlextract import URLExtract
import trafilatura

my_loc = os.path.dirname(__file__)

nlp = spacy.load('en_core_web_lg')

split_mp = {'train': 'train', 'dev':'val', 'test':'test'}

text_processor_wiki = get_text_processor(word_stats='english')
text_processor_twit = get_text_processor(word_stats='twitter')

df_stopwords = set(stopwords.words('english'))

if not os.path.exists('my_code/proc_data/claim_dict.json'):
    cnt = 0
    claim_dict = {}
    with open('data/verified_claims.docs.tsv') as f:
        for line in f:
            if cnt:
                id, claim, title = line.split('\t')
                proc_claim = text_processor_wiki.pre_process_doc(claim)
                proc_title = text_processor_wiki.pre_process_doc(title)

                clean_claim = [word for word in proc_claim if not re.search("[^a-z0-9.,\s]+", word)]
                clean_title = [word for word in proc_title if not re.search("[^a-z0-9.,\s]+", word)]

                claim_dict[id] = {
                    'claim': claim,
                    'title': title,
                    'claim_proc': proc_claim,
                    'title_proc': proc_title,
                    'claim_clean': clean_claim,
                    'title_clean': clean_title,
                }
            
            cnt += 1

    json.dump(claim_dict, open('my_code/proc_data/claim_dict.json','w'))


url_extr = URLExtract()
for split in split_mp:
    data_loc = 'data/'+split
    data_dict = {}
    cnt = 0
    with open(data_loc+'/tweets.queries.tsv','r') as f:
        for line in f:
            if cnt:
                id, tweet = line.strip().split('\t')
                urls = url_extr.find_urls(tweet)

                proc_twit = text_processor_twit.pre_process_doc(tweet)

                clean_twit = [word for word in proc_twit if not re.search("[^a-z0-9.,\s]+", word)]

                data_dict[id] = {
                    'id': id,
                    'text': tweet,
                    'twit_proc': proc_twit,
                    'twit_clean': clean_twit,
                    'urls': urls
                }
            
            cnt += 1

    json.dump(data_dict, open('my_code/proc_data/'+split_mp[split]+'.json','w'))
