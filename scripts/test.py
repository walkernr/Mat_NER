import os
import subprocess
import numpy as np
from pathlib import Path
import torch
from torch import nn
from torch.optim import Adam
from torchtools.optim import RangerLars
from seqeval.metrics import classification_report
from data_utilities import data_format, data_tag, data_split, data_save
from data_tokenizer import MaterialsTextTokenizer
from data_corpus import DataCorpus
from model_crf import CRF
from model_ner import BiLSTM_NER, Transformer_NER
from model_trainer import NERTrainer
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rc('font', family='sans-serif')
ftsz = 28
figw = 16
pparam = {'figure.figsize': (figw, figw),
          'lines.linewidth': 4.0,
          'legend.fontsize': ftsz,
          'axes.labelsize': ftsz,
          'axes.titlesize': ftsz,
          'axes.linewidth': 2.0,
          'xtick.labelsize': ftsz,
          'xtick.major.size': 20,
          'xtick.major.width': 2.0,
          'ytick.labelsize': ftsz,
          'ytick.major.size': 20,
          'ytick.major.width': 2.0,
          'font.size': ftsz}
plt.rcParams.update(pparam)
cm = plt.get_cmap('plasma')

n, m = subprocess.check_output(['stty', 'size']).decode().split()
n, m = int(n), int(m)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 256
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

new_calculation = True
use_history = False

phraser_path = (Path(__file__).parent / '../model/phraser/phraser.pkl').resolve().as_posix()
vector_path = (Path(__file__).parent / '../model/mat2vec/pretrained_embeddings').resolve().as_posix()

tokenizer = MaterialsTextTokenizer(phraser_path=phraser_path)

# corpus parameters
sentence_level = True
cased = True
batch_size = 32

# shared cnn parameters
char_embedding_dim = 38
char_filter = 4
char_kernel = 3
# shared dropouts
embedding_dropout_ratio = 0.5
char_embedding_dropout_ratio = 0.25
cnn_dropout_ratio = 0.25
fc_dropout_ratio = 0.25
# shared attention parameters
attn_heads = 16

# lstm parameters
hidden_dim = 64
lstm_layers = 2
lstm_dropout_ratio = 0.1
attn_dropout_ratio = 0.25
# transformer parameters
hidden_dim = 256
trf_layers = 1
trf_dropout_ratio = 0.25

# parameters for trainer
max_grad_norm = 1.0
n_epoch = 64

# training, validation, testing split
split = (0.8, 0.1, 0.1)

# data_names = ['ner_annotations', 'aunpmorph_annotations_fullparas', 'impurityphase_fullparas', 'doping']
data_names = ['ner_annotations']

for data_name in data_names:
    data_path = (Path(__file__).parent / '../data/{}'.format(data_name)).resolve().as_posix()
    configs = {}
    # configs['_sentence_level_crf_decode_penalties_iobes_{}'.format(seed)] = {'sentence_level': True,
    #                                                                          'format': 'IOBES',
    #                                                                          'use_crf': True,
    #                                                                          'crf_decode': True,
    #                                                                          'crf_penalties': True,
    #                                                                          'lr': {'bilstm': 5e-2, 'transformer': 5e-2}}
    # configs['_sentence_level_crf_decode_penalties_iob2_{}'.format(seed)] = {'sentence_level': True,
    #                                                                         'format': 'IOB2',
    #                                                                         'use_crf': True,
    #                                                                         'crf_decode': True,
    #                                                                         'crf_penalties': True,
    #                                                                         'lr': {'bilstm': 5e-2, 'transformer': 5e-2}}
    # configs['_sentence_level_crf_decode_no_penalties_iobes_{}'.format(seed)] = {'sentence_level': True,
    #                                                                             'format': 'IOBES',
    #                                                                             'use_crf': True,
    #                                                                             'crf_decode': True,
    #                                                                             'crf_penalties': False,
    #                                                                             'lr': {'bilstm': 5e-2, 'transformer': 5e-2}}
    # configs['_sentence_level_crf_decode_no_penalties_iob2_{}'.format(seed)] = {'sentence_level': True,
    #                                                                             'format': 'IOB2',
    #                                                                             'use_crf': True,
    #                                                                             'crf_decode': True,
    #                                                                             'crf_penalties': False,
    #                                                                             'lr': {'bilstm': 5e-2, 'transformer': 5e-2}}
    configs['_sentence_level_crf_no_decode_penalties_iobes_{}'.format(seed)] = {'sentence_level': True,
                                                                                 'format': 'IOBES',
                                                                                 'use_crf': True,
                                                                                 'crf_decode': False,
                                                                                 'crf_penalties': True,
                                                                                 'lr': {'bilstm': 5e-2, 'transformer': 5e-2}}
    configs['_sentence_level_crf_no_decode_penalties_iob2_{}'.format(seed)] = {'sentence_level': True,
                                                                               'format': 'IOB2',
                                                                               'use_crf': True,
                                                                               'crf_decode': False,
                                                                               'crf_penalties': True,
                                                                               'lr': {'bilstm': 5e-2, 'transformer': 5e-2}}
    configs['_sentence_level_crf_no_decode_no_penalties_iobes_{}'.format(seed)] = {'sentence_level': True,
                                                                                   'format': 'IOBES',
                                                                                   'use_crf': True,
                                                                                   'crf_decode': False,
                                                                                   'crf_penalties': False,
                                                                                   'lr': {'bilstm': 5e-2, 'transformer': 5e-2}}
    configs['_sentence_level_crf_no_decode_no_penalties_iob2_{}'.format(seed)] = {'sentence_level': True,
                                                                                  'format': 'IOB2',
                                                                                  'use_crf': True,
                                                                                  'crf_decode': False,
                                                                                  'crf_penalties': False,
                                                                                  'lr': {'bilstm': 5e-2, 'transformer': 5e-2}}
    # configs['_sentence_level_no_crf_iobes_{}'.format(seed)] = {'sentence_level': True,
    #                                                            'format': 'IOBES',
    #                                                            'use_crf': False,
    #                                                            'crf_decode': False,
    #                                                            'crf_penalties': False,
    #                                                            'lr': {'bilstm': 5e-2, 'transformer': 5e-2}}
    # configs['_sentence_level_no_crf_iob2_{}'.format(seed)] = {'sentence_level': True,
    #                                                           'format': 'IOB2',
    #                                                           'use_crf': False,
    #                                                           'crf_decode': False,
    #                                                           'crf_penalties': False,
    #                                                           'lr': {'bilstm': 5e-2, 'transformer': 5e-2}}
              
    for alias, config in configs.items():
        data = data_tag(data_format(data_path, data_name, config['sentence_level']), tag_format=config['format'])
        data_save(data_path, data_name, '_{}'.format(seed), *data_split(data, split, seed))
        corpus = DataCorpus(data_path=data_path, data_name=data_name, alias='_{}'.format(seed), vector_path=vector_path,
                            tokenizer=tokenizer, cased=cased, tag_format=config['format'], batch_size=batch_size, device=device)

        embedding_dim = corpus.embedding_dim
        text_vocab_size = len(corpus.text_field.vocab)
        char_vocab_size = len(corpus.char_field.vocab)
        tag_vocab_size = len(corpus.tag_field.vocab)
        tag_names = corpus.tag_names

        print(m*'-')
        print('vocabularies built')
        print('embedding dimension: {}'.format(embedding_dim))
        print('unique tokens in text vocabulary: {}'.format(text_vocab_size))
        print('unique tokens in char vocabulary: {}'.format(char_vocab_size))
        print('unique tokens in tag vocabulary: {}'.format(tag_vocab_size))
        print('10 most frequent words in text vocabulary: '+(10*'{} ').format(*corpus.text_field.vocab.freqs.most_common(10)))
        print('tags: '+(tag_vocab_size*'{} ').format(*tag_names))
        print(m*'-')

        print('train set: {} sentences'.format(len(corpus.train_set)))
        print('valid set: {} sentences'.format(len(corpus.valid_set)))
        print('test set: {} sentences'.format(len(corpus.valid_set)))
        print(m*'-')

        text_pad_idx = corpus.text_pad_idx
        text_unk_idx = corpus.text_unk_idx
        char_pad_idx = corpus.char_pad_idx
        tag_pad_idx = corpus.tag_pad_idx
        pad_token = corpus.pad_token
        pretrained_embeddings = corpus.text_field.vocab.vectors

        # initialize bilstm
        bilstm = BiLSTM_NER(input_dim=text_vocab_size, embedding_dim=embedding_dim,
                            char_input_dim=char_vocab_size, char_embedding_dim=char_embedding_dim,
                            char_filter=char_filter, char_kernel=char_kernel,
                            hidden_dim=hidden_dim, output_dim=tag_vocab_size,
                            lstm_layers=lstm_layers, attn_heads=attn_heads, use_crf=config['use_crf'],
                            embedding_dropout_ratio=embedding_dropout_ratio, cnn_dropout_ratio=cnn_dropout_ratio, lstm_dropout_ratio=lstm_dropout_ratio,
                            attn_dropout_ratio=attn_dropout_ratio, fc_dropout_ratio=fc_dropout_ratio,
                            tag_names=tag_names, text_pad_idx=text_pad_idx, text_unk_idx=text_unk_idx,
                            char_pad_idx=char_pad_idx, tag_pad_idx=tag_pad_idx, pad_token=pad_token,
                            pretrained_embeddings=pretrained_embeddings, crf_decode=config['crf_decode'], crf_penalties=config['crf_penalties'], tag_format=config['format'])
        # print bilstm information
        print('BiLSTM model initialized with {} trainable parameters'.format(bilstm.count_parameters()))
        print(bilstm)
        print(m*'-')

        # # initialize transformer
        # transformer = Transformer_NER(input_dim=text_vocab_size, embedding_dim=embedding_dim,
        #                             char_input_dim=char_vocab_size, char_embedding_dim=char_embedding_dim,
        #                             char_filter=char_filter, char_kernel=char_kernel,
        #                             hidden_dim=hidden_dim, output_dim=tag_vocab_size,
        #                             trf_layers=trf_layers, attn_heads=attn_heads, use_crf=config['use_crf'],
        #                             embedding_dropout_ratio=embedding_dropout_ratio, cnn_dropout_ratio=cnn_dropout_ratio,
        #                             trf_dropout_ratio=trf_dropout_ratio, fc_dropout_ratio=fc_dropout_ratio,
        #                             tag_names=tag_names, text_pad_idx=text_pad_idx, text_unk_idx=text_unk_idx,
        #                             char_pad_idx=char_pad_idx, tag_pad_idx=tag_pad_idx, pad_token=pad_token,
        #                             pretrained_embeddings=pretrained_embeddings, crf_decode=config['crf_decode'], crf_penalties=config['crf_penalties'], tag_format=config['format'])
        # # print transformer information
        # print('Transformer model initialized with {} trainable parameters'.format(transformer.count_parameters()))
        # print(transformer)
        # print(m*'-')

        # initialize trainer class for bilstm
        bilstm_trainer = NERTrainer(model=bilstm, data=corpus, optimizer_cls=RangerLars, criterion_cls=nn.CrossEntropyLoss,
                                    lr=config['lr']['bilstm'], max_grad_norm=max_grad_norm, device=device)
        # initialize trainer class for transformer
        # transformer_trainer = NERTrainer(model=transformer, data=corpus, optimizer_cls=RangerLars, criterion_cls=nn.CrossEntropyLoss,
        #                                  lr=config['lr']['transformer'], max_grad_norm=max_grad_norm, device=device)

        # bilstm paths
        bilstm_history_path = Path(__file__).parent / '../model/bilstm/history/{}_history.pt'.format(data_name+alias)
        bilstm_test_path = Path(__file__).parent / '../model/bilstm/test/{}_test.pt'.format(data_name+alias)
        bilstm_model_path = Path(__file__).parent / '../model/bilstm/{}_model.pt'.format(data_name+alias)

        # transformer paths
        # transformer_history_path = Path(__file__).parent / '../model/transformer/history/{}_history.pt'.format(data_name+alias)
        # transformer_test_path = Path(__file__).parent / '../model/transformer/test/{}_test.pt'.format(data_name+alias)
        # transformer_model_path = Path(__file__).parent / '../model/transformer/{}_model.pt'.format(data_name+alias)

        # if new_calculation, then train and save the model. otherwise, just load everything from file
        if new_calculation:
            print('training BiLSTM model')
            print(m*'-')
            if use_history:
                if os.path.isfile(bilstm_model_path):
                    print('loading model checkpoint')
                    bilstm_trainer.load_model(model_path=bilstm_model_path)
                    bilstm_trainer.load_history(history_path=bilstm_history_path)
            bilstm_trainer.train(n_epoch=n_epoch)
            bilstm_trainer.load_state_from_cache('best_validation_f1')
            bilstm_trainer.save_model(model_path=bilstm_model_path)
            bilstm_trainer.save_history(history_path=bilstm_history_path)
            print(m*'-')
            # print('training Transformer model')
            # print(m*'-')
            # if use_history:
            #     if os.path.isfile(transformer_model_path):
            #         print('loading model checkpoint')
            #         transformer_trainer.load_model(model_path=transformer_model_path)
            #         transformer_trainer.load_history(history_path=transformer_history_path)
            # transformer_trainer.train(n_epoch=n_epoch)
            # transformer_trainer.load_state_from_cache('best_validation_f1')
            # transformer_trainer.save_model(model_path=transformer_model_path)
            # transformer_trainer.save_history(history_path=transformer_history_path)
            # print(m*'-')
        else:
            print('loading BiLSTM model')
            print(m*'-')
            bilstm_trainer.load_model(model_path=bilstm_model_path)
            bilstm_trainer.load_history(history_path=bilstm_history_path)
            # print('loading Transformer model')
            # print(m*'-')
            # transformer_trainer.load_model(model_path=transformer_model_path)
            # transformer_trainer.load_history(history_path=transformer_history_path)
            print(m*'-')

        # evaluate test set
        print('testing BiLSTM')
        _, prediction_tags, valid_tags = bilstm_trainer.test(bilstm_test_path)
        print(classification_report(valid_tags, prediction_tags, mode=bilstm_trainer.metric_mode,
                                    scheme=bilstm_trainer.metric_scheme))
        print(m*'-')
        # print('testing Transformer')
        # transformer_trainer.test(transformer_test_path)
        # print(m*'-')

        # bilstm_hist = {'training': {}, 'validation': {}}
        # transformer_hist = {'training': {}, 'validation': {}}
        # bilstm_hist_temp = bilstm_trainer.get_history()
        # transformer_hist_temp = transformer_trainer.get_history()
        # for phase in ['training', 'validation']:
        #     for metric in ['loss', 'accuracy_score', 'precision_score', 'recall_score', 'f1_score']:
        #         bilstm_hist[phase][metric] = np.array([bilstm_hist_temp[phase][epoch][metric] for epoch in bilstm_hist_temp[phase].keys()])
        #         transformer_hist[phase][metric] = np.array([transformer_hist_temp[phase][epoch][metric] for epoch in transformer_hist_temp[phase].keys()])

        # hist = {'BiLSTM': (0.3, bilstm_hist),
        #         'Transformer': (0.5, transformer_hist)}
        # hist = {'BiLSTM': (0.3, bilstm_hist)}
        # quant = ['Loss', 'Accuracy Score', 'Precision Score', 'Recall Score', 'F1 Score']
        # phase = ['Training', 'Validation']
        # limits = {'loss': (0.0, 50.0),'accuracy_score': (0.0, 1.0), 'precision_score': (0.0, 1.0), 'recall_score': (0.0, 1.0), 'f1_score': (0.0, 1.0)}
        # # initialize figure and axes
        # fig, axs = plt.subplots(3, 2)
        # # plot losses
        # for i, q in enumerate(quant):
        #     for j, p in enumerate(phase):
        #         qk = q.lower().replace(' ', '_')
        #         pk = p.lower()
        #         ax = axs[i, j]
        #         # remove spines on top and right
        #         ax.spines['right'].set_visible(False)
        #         ax.spines['top'].set_visible(False)
        #         # set axis ticks to left and bottom
        #         ax.xaxis.set_ticks_position('bottom')
        #         ax.yaxis.set_ticks_position('left')
        #         for key, val in hist.items():
        #             ax.plot(val[1][pk][qk].mean(1), color=cm(val[0]), label=key)
        #         ax.set_ylim(*limits[qk])
        #         if 'Loss' in quant[i]:
        #             loc = 'upper right'
        #         else:
        #             loc = 'lower right'
        #         ax.legend(loc=loc, fontsize=ftsz/2)
        #         if i == 2:
        #             ax.set_xlabel('Epoch')
        #         if j == 0:
        #             ax.set_ylabel(quant[i])
        #             if i == 0:
        #                 ax.set_title(phase[j])
        #         if j == 1 and i == 0:
        #             ax.set_title(phase[j])
        # # save figure
        # fig.savefig('history_{}.png'.format(data_name+alias))
        # plt.close()
