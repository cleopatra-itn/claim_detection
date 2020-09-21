import sys
sys.path.append('.')

from my_code.helper_funcs import *
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

                clean_claim = [word for word in proc_claim if not re.search("[^a-z0-9\s]+", word)]
                clean_title = [word for word in proc_title if not re.search("[^a-z0-9\s]+", word)]

                clean_claim = [word for word in clean_claim if len(word) > 2 or word.isnumeric()]
                clean_title = [word for word in clean_title if len(word) > 2 or word.isnumeric()]

                # clean_claim2 = [word for word in clean_claim2 if len(word) > 2]
                # clean_title2 = [word for word in clean_title2 if len(word) > 2]

                clean_claim_nostop = [word for word in clean_claim if word not in df_stopwords]
                clean_title_nostop = [word for word in clean_title if word not in df_stopwords]

                spacy_claim = nlp(" ".join(clean_claim))
                spacy_title = nlp(" ".join(clean_title))
                spacy_claim_nostop = nlp(" ".join(clean_claim_nostop))
                spacy_title_nostop = nlp(" ".join(clean_title_nostop))

                pos_claim, pos_claim_nostop = [], []
                pos_title, pos_title_nostop = [], []
                ner_claim, ner_claim_nostop = [], []
                ner_title, ner_title_nostop = [], []

                for token in spacy_claim:
                    pos_claim.append(token.text+'_'+token.pos_+'_'+token.tag_)
                for token in spacy_title:
                    pos_title.append(token.text+'_'+token.pos_+'_'+token.tag_)

                for token in spacy_claim_nostop:
                    pos_claim_nostop.append(token.text+'_'+token.pos_+'_'+token.tag_)
                for token in spacy_title_nostop:
                    pos_title_nostop.append(token.text+'_'+token.pos_+'_'+token.tag_)

                for ent in spacy_claim.ents:
                    ner_claim.append({
                            'text': ent.text,
                            'label': ent.label_,
                            'start': ent.start_char,
                            'end': ent.end_char,
                        })

                for ent in spacy_title.ents:
                    ner_title.append({
                            'text': ent.text,
                            'label': ent.label_,
                            'start': ent.start_char,
                            'end': ent.end_char,
                        })

                for ent in spacy_claim_nostop.ents:
                    ner_claim_nostop.append({
                            'text': ent.text,
                            'label': ent.label_,
                            'start': ent.start_char,
                            'end': ent.end_char,
                        })

                for ent in spacy_title_nostop.ents:
                    ner_title_nostop.append({
                            'text': ent.text,
                            'label': ent.label_,
                            'start': ent.start_char,
                            'end': ent.end_char,
                        })
                
                claim_dict[id] = {
                    'claim': claim,
                    'title': title,
                    'claim_proc': proc_claim,
                    'title_proc': proc_title,
                    'claim_clean': clean_claim,
                    'title_clean': clean_title,
                    'claim_clean_nostop': clean_claim_nostop,
                    'title_clean_nostop': clean_title_nostop,
                    'pos_claim': pos_claim,
                    'pos_title': pos_title,
                    'pos_claim_nostop': pos_claim_nostop,
                    'pos_title_nostop': pos_title_nostop,
                    'ner_claim': ner_claim,
                    'ner_title': ner_title,
                    'ner_claim_nostop': ner_claim_nostop,
                    'ner_title_nostop': ner_title_nostop
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

                proc_wiki = text_processor_wiki.pre_process_doc(tweet)
                proc_twit = text_processor_twit.pre_process_doc(tweet)

                clean_wiki = [word for word in proc_wiki if not re.search("[^a-z0-9\s]+", word)]
                clean_twit = [word for word in proc_twit if not re.search("[^a-z0-9\s]+", word)]

                clean_wiki = [word for word in clean_wiki if len(word) > 2 or word.isnumeric()]
                clean_twit = [word for word in clean_twit if len(word) > 2 or word.isnumeric()]

                clean_wiki_nostop = [word for word in clean_wiki if word not in df_stopwords]
                clean_twit_nostop = [word for word in clean_twit if word not in df_stopwords]

                spacy_wiki = nlp(" ".join(clean_wiki))
                spacy_twit = nlp(" ".join(clean_twit))
                spacy_wiki_nostop = nlp(" ".join(clean_wiki_nostop))
                spacy_twit_nostop = nlp(" ".join(clean_twit_nostop))

                pos_wiki, pos_wiki_nostop = [], []
                pos_twit, pos_twit_nostop = [], []
                ner_wiki, ner_wiki_nostop = [], []
                ner_twit, ner_twit_nostop = [], []

                for token in spacy_wiki:
                    pos_wiki.append(token.text+'_'+token.pos_+'_'+token.tag_)
                for token in spacy_twit:
                    pos_twit.append(token.text+'_'+token.pos_+'_'+token.tag_)

                for token in spacy_wiki_nostop:
                    pos_wiki_nostop.append(token.text+'_'+token.pos_+'_'+token.tag_)
                for token in spacy_twit_nostop:
                    pos_twit_nostop.append(token.text+'_'+token.pos_+'_'+token.tag_)

                for ent in spacy_wiki.ents:
                    ner_wiki.append({
                            'text': ent.text,
                            'label': ent.label_,
                            'start': ent.start_char,
                            'end': ent.end_char,
                        })

                for ent in spacy_twit.ents:
                    ner_twit.append({
                            'text': ent.text,
                            'label': ent.label_,
                            'start': ent.start_char,
                            'end': ent.end_char,
                        })

                for ent in spacy_wiki_nostop.ents:
                    ner_wiki_nostop.append({
                            'text': ent.text,
                            'label': ent.label_,
                            'start': ent.start_char,
                            'end': ent.end_char,
                        })

                for ent in spacy_twit_nostop.ents:
                    ner_twit_nostop.append({
                            'text': ent.text,
                            'label': ent.label_,
                            'start': ent.start_char,
                            'end': ent.end_char,
                        })

                data_dict[id] = {
                    'id': id,
                    'text': tweet,
                    'wiki_proc': proc_wiki,
                    'twit_proc': proc_twit,
                    'wiki_clean': clean_wiki,
                    'twit_clean': clean_twit,
                    'wiki_clean_nostop': clean_wiki_nostop,
                    'twit_clean_nostop': clean_twit_nostop,
                    'pos_wiki': pos_wiki,
                    'pos_twit': pos_twit,
                    'pos_wiki_nostop': pos_wiki_nostop,
                    'pos_twit_nostop': pos_twit_nostop,
                    'ner_wiki': ner_wiki,
                    'ner_twit': ner_twit,
                    'ner_wiki_nostop': ner_wiki_nostop,
                    'ner_twit_nostop': ner_twit_nostop,
                    'urls': urls
                }
            
            cnt += 1

    json.dump(data_dict, open('my_code/proc_data/'+split_mp[split]+'.json','w'))
