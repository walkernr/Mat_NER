import os
import argparse
from pathlib import Path
import numpy as np
from seqeval.scheme import IOB1, IOB2, IOBES
from seqeval.metrics import classification_report
from data_utilities import collect_abstracts, split_abstracts, format_abstracts, tag_abstracts, save_tagged_splits
from data_tokenizer import MaterialsTextTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dv', '--device', help='computation device for model (e.g. cpu, gpu:0, gpu:1)', type=str, default='gpu:0')
    parser.add_argument('-sd', '--seeds', help='comma-separated seeds for data shuffling and model initialization (e.g. 1,2,3 or 2,4,8)', type=str, default='256')
    parser.add_argument('-ts', '--tag_schemes', help='comma-separated tagging schemes to be considered (e.g. IOB1,IOB2,IOBES)', type=str, default='IOBES')
    parser.add_argument('-st', '--splits', help='comma-separated training splits to be considered, in percent (e.g. 80). test split will always be 10%% and the validation split will be 1/8 of the training split', type=str, default='80')
    parser.add_argument('-ds', '--datasets', help='comma-separated datasets to be considered (e.g. solid_state,doping)', type=str, default='solid_state')
    parser.add_argument('-sl', '--sentence_level', help='switch for sentence-level learning instead of paragraph-level', action='store_true')
    parser.add_argument('-bs', '--batch_size', help='number of samples in each batch', type=int, default=32)
    parser.add_argument('-ne', '--n_epochs', help='number of training epochs', type=int, default=64)
    parser.add_argument('-lr', '--learning_rate', help='optimizer learning rate', type=float, default=5e-2)
    parser.add_argument('-km', '--keep_model', help='switch for saving the best model parameters to disk', action='store_true')
    args = parser.parse_args()
    return args.device, args.seeds, args.tag_schemes, args.splits, args.datasets, args.sentence_level, args.batch_size, args.n_epochs, args.learning_rate, args.keep_model


if __name__ == '__main__':
    device, seeds, tag_schemes, splits, datasets, sentence_level, batch_size, n_epochs, lr, keep_model = parse_args()
    m = 80
    if 'gpu' in device:
        gpu = True
        try:
            d, n = device.split(':')
        except:
            print('ValueError: Improper device format in command-line argument')
        device = 'cuda'
    else:
        gpu = False
    if gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(n)
    from data_corpus import DataCorpus
    import torch
    from torch import nn
    from torch.optim import Adam
    from torchtools.optim import RangerLars
    from model_crf import CRF
    from model_ner import BiLSTM_NER, Transformer_NER
    from model_trainer import NERTrainer

    torch.device('cuda' if gpu else 'cpu')
    torch.backends.cudnn.deterministic = True

    seeds = [int(seed) for seed in seeds.split(',')]
    tag_schemes = [str(tag_scheme) for tag_scheme in tag_schemes.split(',')]
    splits = [int(split) for split in splits.split(',')]
    datasets = [str(dataset) for dataset in datasets.split(',')]

    phraser_path = (Path(__file__).parent / '../model/phraser/phraser.pkl').resolve().as_posix()
    vector_path = (Path(__file__).parent / '../model/mat2vec/pretrained_embeddings').resolve().as_posix()
    schemes = {'IOB1': IOB1, 'IOB2': IOB2, 'IOBES': IOBES}

    tokenizer = MaterialsTextTokenizer(phraser_path=phraser_path)

    # corpus parameters
    cased = True
    # cnn parameters
    char_embedding_dim = 64
    char_filter = 4
    char_kernel = 2
    # dropouts
    embedding_dropout_ratio = 0.5
    char_embedding_dropout_ratio = 0.25
    cnn_dropout_ratio = 0.25
    fc_dropout_ratio = 0.25
    # attention parameters
    attn_heads = 16
    # lstm parameters
    hidden_dim = 64
    lstm_layers = 2
    lstm_dropout_ratio = 0.1
    attn_dropout_ratio = 0.25
    # parameters for trainer
    max_grad_norm = 1.0
    n_epoch = 64

    for seed in seeds:
        for tag_scheme in tag_schemes:
            for split in splits:
                for dataset in datasets:
                    torch.manual_seed(seed)
                    data_path = (Path(__file__).parent / '../data/{}'.format(dataset)).resolve().as_posix()
                    alias = '{}_{}_crf_{}_{}_{}'.format(dataset, 'sentence' if sentence_level else 'paragraph', tag_scheme.lower(), seed, split)
                    # bilstm paths
                    bilstm_history_path = (Path(__file__).parent / '../model/bilstm/history/{}_history.pt'.format(alias)).resolve().as_posix()
                    bilstm_test_path = (Path(__file__).parent / '../model/bilstm/test/{}_test.pt'.format(alias)).resolve().as_posix()
                    bilstm_model_path = (Path(__file__).parent / '../model/bilstm/{}_model.pt'.format(alias)).resolve().as_posix()
                    if os.path.exists(bilstm_test_path):
                        print('already calculated {}, skipping'.format(alias))
                    else:
                        try:
                            data = tag_abstracts(format_abstracts(split_abstracts(collect_abstracts(data_path, dataset), (0.1, split/800, split/100), seed), seed, sentence_level), tag_scheme)
                            save_tagged_splits(data_path, dataset, '_{}_{}_{}_{}'.format('sentence' if sentence_level else 'paragraph', tag_scheme.lower(), seed, split), data)
                            corpus = DataCorpus(data_path=data_path, data_name=dataset, alias='_{}_{}_{}_{}'.format('sentence' if sentence_level else 'paragraph', tag_scheme.lower(), seed, split), vector_path=vector_path,
                                                tokenizer=tokenizer, cased=cased, tag_scheme=tag_scheme, batch_size=batch_size, device=device)
                            
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

                            print('train set: {} {}s'.format(len(corpus.train_set), 'sentence' if sentence_level else 'paragraph'))
                            print('valid set: {} {}s'.format(len(corpus.valid_set), 'sentence' if sentence_level else 'paragraph'))
                            print('test set: {} {}s'.format(len(corpus.test_set), 'sentence' if sentence_level else 'paragraph'))
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
                                                lstm_layers=lstm_layers, attn_heads=attn_heads, use_crf=True,
                                                embedding_dropout_ratio=embedding_dropout_ratio, cnn_dropout_ratio=cnn_dropout_ratio, lstm_dropout_ratio=lstm_dropout_ratio,
                                                attn_dropout_ratio=attn_dropout_ratio, fc_dropout_ratio=fc_dropout_ratio,
                                                tag_names=tag_names, text_pad_idx=text_pad_idx, text_unk_idx=text_unk_idx,
                                                char_pad_idx=char_pad_idx, tag_pad_idx=tag_pad_idx, pad_token=pad_token,
                                                pretrained_embeddings=pretrained_embeddings, tag_scheme=tag_scheme)
                            # print bilstm information
                            print('BiLSTM model initialized with {} trainable parameters'.format(bilstm.count_parameters()))
                            print(bilstm)
                            print(m*'-')


                            # initialize trainer class for bilstm
                            bilstm_trainer = NERTrainer(model=bilstm, data=corpus, optimizer_cls=RangerLars, criterion_cls=nn.CrossEntropyLoss,
                                                        lr=lr, max_grad_norm=max_grad_norm, device=device)
                            
                            print('training BiLSTM model')
                            print(m*'-')
                            bilstm_trainer.train(n_epoch=n_epoch)
                            bilstm_trainer.load_state_from_cache('best_validation_f1')
                            if keep_model:
                                bilstm_trainer.save_model(model_path=bilstm_model_path)
                            bilstm_trainer.save_history(history_path=bilstm_history_path)
                            print(m*'-')

                            print('testing BiLSTM')
                            _, _, _, labels, predictions = bilstm_trainer.test(bilstm_test_path)
                            print(classification_report(labels, predictions, mode=bilstm_trainer.metric_mode,
                                                        scheme=bilstm_trainer.metric_scheme))
                            print(m*'-')
                        except:
                            print('error calculating {}'.format(alias))