import sys
sys.path.append('.')
from my_code.helper_funcs import *
import pickle
import spacy
import re, os
from nltk.corpus import stopwords
import json
from urlextract import URLExtract

my_loc = os.path.dirname(__file__)

nlp = spacy.load('en_core_web_lg')

split_mp = {'training': 'train', 'dev':'val', 'test': 'test'}

text_processor_twit = get_text_processor(word_stats='twitter')

df_stopwords = set(stopwords.words('english'))
url_extr = URLExtract()

for split in split_mp:
    tr_file = open('data/%s.tsv'%(split), 'r')
    data_dict = {}
    cnt = 0
    for line in tr_file:
        if cnt:
            if split != 'test':
                topic, id, link, content, claim, worthy = line.strip().split('\t')
            else:
                topic, id, link, content = line.strip().split('\t')
                claim, worthy = 0, 0

            urls = url_extr.find_urls(content)

            proc_twit = text_processor_twit.pre_process_doc(content)

            clean_twit = [word for word in proc_twit if not re.search("[^a-z0-9\s]+", word)]

            clean_twit = [word for word in clean_twit if len(word) > 2 or word.isnumeric()]

            clean_twit_nostop = [word for word in clean_twit if word not in df_stopwords]

            spacy_twit = nlp(" ".join(clean_twit))
            spacy_twit_nostop = nlp(" ".join(clean_twit_nostop))

            pos_twit, pos_twit_nostop = [], []
            ner_twit, ner_twit_nostop = [], []


            for token in spacy_twit:
                pos_twit.append(token.text+'_'+token.pos_+'_'+token.tag_)

            for token in spacy_twit_nostop:
                pos_twit_nostop.append(token.text+'_'+token.pos_+'_'+token.tag_)

            for ent in spacy_twit.ents:
                ner_twit.append({
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
                'link': link,
                'text': content,
                'twit_proc': proc_twit,
                'twit_clean': clean_twit,
                'twit_clean_nostop': clean_twit_nostop,
                'pos_twit': pos_twit,
                'pos_twit_nostop': pos_twit_nostop,
                'ner_twit': ner_twit,
                'ner_twit_nostop': ner_twit_nostop,
                'claim': claim,
                'worthy': worthy,
                'urls': urls
            }

        cnt += 1


    # Data Stats, PoS and NER
    ner_dict, ner_ns_dict = {'1': {}, '0':{}}, {'1': {}, '0':{}}
    pos_dict, pos_ns_dict = {'1': {}, '0':{}}, {'1': {}, '0':{}}

    for id in data_dict:
        lab = str(data_dict[id]['worthy'])
        for ptag in data_dict[id]['pos_twit']:
            tg = ptag.split('_')[1]
            pos_dict[lab][tg] = 1 if tg not in pos_dict[lab] else pos_dict[lab][tg]+1

        for ptag in data_dict[id]['pos_twit_nostop']:
            tg = ptag.split('_')[1]
            pos_ns_dict[lab][tg] = 1 if tg not in pos_ns_dict[lab] else pos_ns_dict[lab][tg]+1

        for netag in data_dict[id]['ner_twit']:
            tg = netag['label']
            ner_dict[lab][tg] = 1 if tg not in ner_dict[lab] else ner_dict[lab][tg]+1

        for netag in data_dict[id]['ner_twit_nostop']:
            tg =  netag['label']
            ner_ns_dict[lab][tg] = 1 if tg not in ner_ns_dict[lab] else ner_ns_dict[lab][tg]+1


    pos_dict['0'] = {k: v for k, v in sorted(pos_dict['0'].items(), key=lambda item: item[1])}
    pos_dict['1'] = {k: v for k, v in sorted(pos_dict['1'].items(), key=lambda item: item[1])}
    pos_ns_dict['0'] = {k: v for k, v in sorted(pos_ns_dict['0'].items(), key=lambda item: item[1])}
    pos_ns_dict['1'] = {k: v for k, v in sorted(pos_ns_dict['1'].items(), key=lambda item: item[1])}
    ner_dict['0'] = {k: v for k, v in sorted(ner_dict['0'].items(), key=lambda item: item[1])}
    ner_dict['1'] = {k: v for k, v in sorted(ner_dict['1'].items(), key=lambda item: item[1])}
    ner_ns_dict['0'] = {k: v for k, v in sorted(ner_ns_dict['0'].items(), key=lambda item: item[1])}
    ner_ns_dict['1'] = {k: v for k, v in sorted(ner_ns_dict['1'].items(), key=lambda item: item[1])}
    print(pos_dict)
    print(pos_ns_dict)
    print(ner_dict)
    print(ner_ns_dict)

    # json.dump(data_dict, open(my_loc+'/proc_data/%s_data.json'%(split_mp[split]), 'w', encoding='utf-8'))
