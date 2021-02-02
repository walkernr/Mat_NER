import json
import random
import numpy as np


def data_format(data_path, data_name):
    data = []
    with open(data_path+'/original/'+data_name+'.json', 'r') as f:
        for line in f:
            d = json.loads(line)
            dat = [{'text': [token['text'] for token in sentence], 'annotation': [token['annotation'] for token in sentence]} for sentence in d['tokens']]
            data.extend(dat)
    return data


def data_tag(data, format):
    for dat in data:
        annotation = dat['annotation']
        dat['tag'] = []
        for i in range(len(annotation)):
            if format == 'IOB2':
                if annotation[i] == None:
                    dat['tag'].append('O')
                elif i == 0:
                    dat['tag'].append('B-'+annotation[i])
                elif i > 0:
                    if annotation[i-1] == annotation[i]:
                        dat['tag'].append('I-'+annotation[i])
                    else:
                        dat['tag'].append('B-'+annotation[i])
            if format == 'BILOU':
                if annotation[i] == None:
                    dat['tag'].append('O')
                elif i == 0:
                    if annotation[i+1] == annotation[i]:
                        dat['tag'].append('B-'+annotation[i])
                    else:
                        dat['tag'].append('U-'+annotation[i])
                elif i > 0 and i < len(annotation)-1:
                    if annotation[i-1] != annotation[i] and annotation[i+1] == annotation[i]:
                        dat['tag'].append('B-'+annotation[i])
                    elif annotation[i-1] == annotation[i] and annotation[i+1] == annotation[i]:
                        dat['tag'].append('I-'+annotation[i])
                    elif annotation[i-1] == annotation[i] and annotation[i+1] != annotation[i]:
                        dat['tag'].append('L-'+annotation[i])
                    if annotation[i-1] != annotation[i] and annotation[i+1] != annotation[i]:
                        dat['tag'].append('U-'+annotation[i])
                elif i == len(annotation):
                    if annotation[i-1] == annotation[i]:
                        dat['tag'].append('L-'+annotation[i])
                    if annotation[i-1] != annotation[i]:
                        dat['tag'].append('U-'+annotation[i])
            if format == 'BIOES':
                if annotation[i] == None:
                    dat['tag'].append('O')
                elif i == 0:
                    if annotation[i+1] == annotation[i]:
                        dat['tag'].append('B-'+annotation[i])
                    else:
                        dat['tag'].append('S-'+annotation[i])
                elif i > 0 and i < len(annotation)-1:
                    if annotation[i-1] != annotation[i] and annotation[i+1] == annotation[i]:
                        dat['tag'].append('B-'+annotation[i])
                    elif annotation[i-1] == annotation[i] and annotation[i+1] == annotation[i]:
                        dat['tag'].append('I-'+annotation[i])
                    elif annotation[i-1] == annotation[i] and annotation[i+1] != annotation[i]:
                        dat['tag'].append('E-'+annotation[i])
                    if annotation[i-1] != annotation[i] and annotation[i+1] != annotation[i]:
                        dat['tag'].append('S-'+annotation[i])
                elif i == len(annotation):
                    if annotation[i-1] == annotation[i]:
                        dat['tag'].append('E-'+annotation[i])
                    if annotation[i-1] != annotation[i]:
                        dat['tag'].append('S-'+annotation[i])
    return data


def data_split(data, splits, seed):
    splits = (np.cumsum(splits)*len(data)).astype(np.uint16)
    random.Random(seed).shuffle(data)
    train_set = data[:splits[0]]
    valid_set = data[splits[0]:splits[1]]
    test_set = data[splits[1]:splits[2]]
    return train_set, valid_set, test_set


def data_save(data_path, data_name, alias, train_set, valid_set, test_set):
    data = [train_set, valid_set, test_set]
    data_names = [data_name+alias+'_{}.tsv'.format(n) for n in ['train', 'valid', 'test']]
    for dat, dat_name in zip(data, data_names):
        with open(data_path+'/split/'+dat_name, 'w') as f:
            for d in dat:
                for txt, tg in zip(d['text'], d['tag']):
                    f.write('{}\t{}\n'.format(txt, tg))
                f.write('\n')              
