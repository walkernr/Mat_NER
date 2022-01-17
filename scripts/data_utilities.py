import json
import random
import numpy as np


def collect_abstracts(data_path, data_name):
    identifiers = []
    data = []
    with open(data_path+'/original/'+data_name+'.json', 'r') as f:
        for line in f:
            d = json.loads(line)
            if 'solid_state' in data_name:
                identifier = d['doi']
            elif 'aunp' in data_name:
                identifier = d['meta']['doi']+'/'+str(d['meta']['par'])
            elif 'doping' in data_name:
                identifier = d['text']
            else:
                try:
                    identifier = d['doi']
                except:
                    identifier = d['meta']['doi']+'/'+str(d['meta']['par'])
            if identifier in identifiers:
                pass
            else:
                identifiers.append(identifier)
                data.append(d)
    return data


def split_abstracts(data, splits, seed):
    if seed:
        random.Random(seed).shuffle(data)
    else:
        random.shuffle(data)
    splits = (np.cumsum(splits)*len(data)).astype(np.uint16)
    test_set = data[:splits[0]]
    valid_set = data[splits[0]:splits[1]]
    train_set = data[splits[1]:splits[2]]
    data_split = {'test': test_set, 'valid': valid_set, 'train': train_set}
    return data_split


def format_abstracts(data_split, sentence_level=True):
    data_fmt = {key: [] for key in data_split.keys()}
    for split in data_split.keys():
        for d in data_split[split]:
            # if split == 'test':
            #     print(d['doi'])
            if sentence_level:
                dat = [{'text': [token['text'] for token in sentence], 'annotation': [token['annotation'] for token in sentence]} for sentence in d['tokens']]
            else:
                dat = [{key : [token[key] for sentence in d['tokens'] for token in sentence] for key in ['text', 'annotation']}]
            data_fmt[split].extend(dat)
    return data_fmt


def label_abstracts(data_fmt, scheme):
    for split in data_fmt.keys():
        for dat in data_fmt[split]:
            annotation = dat['annotation']
            dat['label'] = []
            for i in range(len(annotation)):
                if scheme == 'IOB1':
                    if annotation[i] in [None, 'PVL', 'PUT']:
                        dat['label'].append('O')
                    elif i == 0 and len(annotation) > 1:
                        if annotation[i+1] == annotation[i]:
                            dat['label'].append('B-'+annotation[i])
                        else:
                            dat['label'].append('I-'+annotation[i])
                    elif i == 0 and len(annotation) == 1:
                        dat['label'].append('I'-annotation[i])
                    elif i > 0:
                        if annotation[i-1] == annotation[i]:
                            dat['label'].append('I-'+annotation[i])
                        else:
                            if annotation[i+1] == annotation[i]:
                                dat['label'].append('B-'+annotation[i])
                            else:
                                dat['label'].append('I-'+annotation[i])
                elif scheme == 'IOB2':
                    if annotation[i] in [None, 'PVL', 'PUT']:
                        dat['label'].append('O')
                    elif i == 0:
                        dat['label'].append('B-'+annotation[i])
                    elif i > 0:
                        if annotation[i-1] == annotation[i]:
                            dat['label'].append('I-'+annotation[i])
                        else:
                            dat['label'].append('B-'+annotation[i])
                elif scheme == 'IOBES':
                    if annotation[i] in [None, 'PVL', 'PUT']:
                        dat['label'].append('O')
                    elif i == 0 and len(annotation) > 1:
                        if annotation[i+1] == annotation[i]:
                            dat['label'].append('B-'+annotation[i])
                        else:
                            dat['label'].append('S-'+annotation[i])
                    elif i == 0 and len(annotation) == 1:
                        dat['label'].append('S-'+annotation[i])
                    elif i > 0 and i < len(annotation)-1:
                        if annotation[i-1] != annotation[i] and annotation[i+1] == annotation[i]:
                            dat['label'].append('B-'+annotation[i])
                        elif annotation[i-1] == annotation[i] and annotation[i+1] == annotation[i]:
                            dat['label'].append('I-'+annotation[i])
                        elif annotation[i-1] == annotation[i] and annotation[i+1] != annotation[i]:
                            dat['label'].append('E-'+annotation[i])
                        if annotation[i-1] != annotation[i] and annotation[i+1] != annotation[i]:
                            dat['label'].append('S-'+annotation[i])
                    elif i == len(annotation)-1:
                        if annotation[i-1] == annotation[i]:
                            dat['label'].append('E-'+annotation[i])
                        if annotation[i-1] != annotation[i]:
                            dat['label'].append('S-'+annotation[i])
    return data_fmt


def save_labeled_splits(data_path, data_name, alias, data_fmt):
    for split in data_fmt.keys():
        dat_name = data_name+alias+'_{}.tsv'.format(split)
        with open(data_path+'/split/'+dat_name, 'w') as f:
            for d in data_fmt[split]:
                for txt, tg in zip(d['text'], d['label']):
                    f.write('{}\t{}\n'.format(txt, tg))
                f.write('\n')
