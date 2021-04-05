import json
import random
import numpy as np


def data_format(data_path, data_name, sentence_level=True):
    '''
    formats the raw data such that each sample is separated into lists of tokens and annotations

    data_path: root directory of the data
    data_name: name of the data file (assumes json)
    sentence_level: switch for splitting samples (abstracts) into sentences
    '''
    data = []
    with open(data_path+'/original/'+data_name+'.json', 'r') as f:
        for line in f:
            d = json.loads(line)
            if sentence_level:
                dat = [{'text': [token['text'] for token in sentence], 'annotation': [token['annotation'] for token in sentence]} for sentence in d['tokens']]
            else:
                dat = [{key : [token[key] for sentence in d['tokens'] for token in sentence] for key in ['text', 'annotation']}]
            data.extend(dat)
    return data


def data_tag(data, tag_format):
    '''
    tags the data

    tag_format: IOB1, IOB2, or IOBES
    '''
    for dat in data:
        annotation = dat['annotation']
        dat['tag'] = []
        for i in range(len(annotation)):
            if tag_format == 'IOB1':
                if annotation[i] in [None, 'PVL', 'PUT']:
                    dat['tag'].append('O')
                elif i == 0:
                    if annotation[i+1] == annotation[i]:
                        dat['tag'].append('B-'+annotation[i])
                    else:
                        dat['tag'].append('I-'+annotation[i])
                elif i > 0:
                    if annotation[i-1] == annotation[i]:
                        dat['tag'].append('I-'+annotation[i])
                    else:
                        if annotation[i+1] == annotation[i]:
                            dat['tag'].append('B-'+annotation[i])
                        else:
                            dat['tag'].append('I-'+annotation[i])
            elif tag_format == 'IOB2':
                if annotation[i] in [None, 'PVL', 'PUT']:
                    dat['tag'].append('O')
                elif i == 0:
                    dat['tag'].append('B-'+annotation[i])
                elif i > 0:
                    if annotation[i-1] == annotation[i]:
                        dat['tag'].append('I-'+annotation[i])
                    else:
                        dat['tag'].append('B-'+annotation[i])
            elif tag_format == 'IOBES':
                if annotation[i] in [None, 'PVL', 'PUT']:
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
                elif i == len(annotation)-1:
                    if annotation[i-1] == annotation[i]:
                        dat['tag'].append('E-'+annotation[i])
                    if annotation[i-1] != annotation[i]:
                        dat['tag'].append('S-'+annotation[i])
    return data


def data_split(data, splits, seed):
    '''
    splits the data into training, validation, and test sets

    data: list of tagged samples
    splits: (train, validation, test) proportional splits
    seed: random seed for shuffling
    '''
    splits = (np.cumsum(splits)*len(data)).astype(np.uint16)
    np.random.seed(seed)
    np.random.shuffle(data)
    # random.Random(seed).shuffle(data)
    test_set = data[:splits[0]]
    valid_set = data[splits[0]:splits[1]]
    train_set = data[splits[1]:splits[2]]
    return train_set, valid_set, test_set


def data_save(data_path, data_name, alias, train_set, valid_set, test_set):
    '''
    formats the raw data such that each sample is separated into lists of tokens and annotations
    
    data_path: root directory of the data
    data_name: name of the data file (assumes json)
    alias: alias to append to file names
    train_set: list of training samples
    valid_set: list of validation samples
    test_set: list of test samples
    '''
    data = [train_set, valid_set, test_set]
    data_names = [data_name+alias+'_{}.tsv'.format(n) for n in ['train', 'valid', 'test']]
    for dat, dat_name in zip(data, data_names):
        with open(data_path+'/split/'+dat_name, 'w') as f:
            for d in dat:
                for txt, tg in zip(d['text'], d['tag']):
                    f.write('{}\t{}\n'.format(txt, tg))
                f.write('\n')              
