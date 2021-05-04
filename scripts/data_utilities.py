import json
import random
import numpy as np


def collect_abstracts(data_path, data_name):
    identifiers = []
    data = []
    with open(data_path+'/original/'+data_name+'.json', 'r') as f:
        for line in f:
            d = json.loads(line)
            if data_name == 'solid_state':
                identifier = d['doi']
            elif data_name in ['doping', 'aunpmorph']:
                identifier = d['text']
            if identifier in identifiers:
                pass
            else:
                identifiers.append(identifier)
                data.append(d)
    return data


def split_abstracts(data, splits, seed):
    random.Random(seed).shuffle(data)
    splits = (np.cumsum(splits)*len(data)).astype(np.uint16)
    test_set = data[:splits[0]]
    valid_set = data[splits[0]:splits[1]]
    train_set = data[splits[1]:splits[2]]
    data_split = {'test': test_set, 'valid': valid_set, 'train': train_set}
    return data_split


def format_abstracts(data_split, seed, sentence_level=True):
    data_fmt = {key: [] for key in data_split.keys()}
    for split in data_split.keys():
        for d in data_split[split]:
            if sentence_level:
                dat = [{'text': [token['text'] for token in sentence], 'annotation': [token['annotation'] for token in sentence]} for sentence in d['tokens']]
            else:
                dat = [{key : [token[key] for sentence in d['tokens'] for token in sentence] for key in ['text', 'annotation']}]
            data_fmt[split].extend(dat)
        random.Random(seed).shuffle(data_fmt[split])
    return data_fmt


def tag_abstracts(data_fmt, tag_scheme):
    for split in data_fmt.keys():
        for dat in data_fmt[split]:
            annotation = dat['annotation']
            dat['tag'] = []
            for i in range(len(annotation)):
                if tag_scheme == 'IOB1':
                    if annotation[i] in [None, 'PVL', 'PUT']:
                        dat['tag'].append('O')
                    elif i == 0 and len(annotation) > 1:
                        if annotation[i+1] == annotation[i]:
                            dat['tag'].append('B-'+annotation[i])
                        else:
                            dat['tag'].append('I-'+annotation[i])
                    elif i == 0 and len(annotation) == 1:
                        dat['tag'].append('I'-annotation[i])
                    elif i > 0:
                        if annotation[i-1] == annotation[i]:
                            dat['tag'].append('I-'+annotation[i])
                        else:
                            if annotation[i+1] == annotation[i]:
                                dat['tag'].append('B-'+annotation[i])
                            else:
                                dat['tag'].append('I-'+annotation[i])
                elif tag_scheme == 'IOB2':
                    if annotation[i] in [None, 'PVL', 'PUT']:
                        dat['tag'].append('O')
                    elif i == 0:
                        dat['tag'].append('B-'+annotation[i])
                    elif i > 0:
                        if annotation[i-1] == annotation[i]:
                            dat['tag'].append('I-'+annotation[i])
                        else:
                            dat['tag'].append('B-'+annotation[i])
                elif tag_scheme == 'IOBES':
                    if annotation[i] in [None, 'PVL', 'PUT']:
                        dat['tag'].append('O')
                    elif i == 0 and len(annotation) > 1:
                        if annotation[i+1] == annotation[i]:
                            dat['tag'].append('B-'+annotation[i])
                        else:
                            dat['tag'].append('S-'+annotation[i])
                    elif i == 0 and len(annotation) == 1:
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
    return data_fmt


def save_tagged_splits(data_path, data_name, alias, data_fmt):
    for split in data_fmt.keys():
        dat_name = data_name+alias+'_{}.tsv'.format(split)
        with open(data_path+'/split/'+dat_name, 'w') as f:
            for d in data_fmt[split]:
                for txt, tg in zip(d['text'], d['tag']):
                    f.write('{}\t{}\n'.format(txt, tg))
                f.write('\n')
