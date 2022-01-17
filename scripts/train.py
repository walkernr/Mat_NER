import os
import argparse
from pathlib import Path
import numpy as np
from seqeval.scheme import IOB1, IOB2, IOBES
from seqeval.metrics import classification_report
from data_utilities import collect_abstracts, split_abstracts, format_abstracts, label_abstracts, save_labeled_splits
from data_tokenizer import MaterialsTextTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dv', '--device', help='computation device for model (e.g. cpu, gpu:0, gpu:1)', type=str, default='gpu:0')
    parser.add_argument('-sd', '--seeds', help='comma-separated seeds for data shuffling and model initialization (e.g. 1,2,3 or 2,4,8)', type=str, default='256')
    parser.add_argument('-ts', '--tag_schemes', help='comma-separated tagging schemes to be considered (e.g. iob1,iob2,iobes)', type=str, default='iobes')
    parser.add_argument('-st', '--splits', help='comma-separated training splits to be considered, in percent (e.g. 80). test split will always be 10%% and the validation split will be 1/8 of the training split', type=str, default='80')
    parser.add_argument('-ds', '--datasets', help='comma-separated datasets to be considered (e.g. solid_state,doping)', type=str, default='solid_state')
    parser.add_argument('-sl', '--sentence_level', help='switch for sentence-level learning instead of paragraph-level', action='store_true')
    parser.add_argument('-bs', '--batch_size', help='number of samples in each batch', type=int, default=10)
    parser.add_argument('-on', '--optimizer_name', help='name of optimizer, add "_lookahead" to implement lookahead on top of optimizer (not recommended for ranger or rangerlars)', type=str, default='lamb')
    parser.add_argument('-wd', '--weight_decay', help='weight decay for optimizer (excluding bias, gamma, and beta)', type=float, default=0.0)
    parser.add_argument('-ne', '--n_epochs', help='number of training epochs', type=int, default=16)
    parser.add_argument('-eu', '--embedding_unfreeze', help='epoch (index) at which mat2vec embeddings are unfrozen', type=int, default=16)
    parser.add_argument('-lr', '--learning_rate', help='optimizer learning rate', type=float, default=3e-2)
    parser.add_argument('-sf', '--scheduling_function', help='function for learning rate scheduler (linear, exponential, or cosine)', type=str, default='cosine')
    parser.add_argument('-su', '--scheduling_unlock', help='epoch (index) at which scheduler starts', type=int, default=0)
    parser.add_argument('-km', '--keep_model', help='switch for saving the best model parameters to disk', action='store_true')
    args = parser.parse_args()
    return args.device, args.seeds, args.tag_schemes, args.splits, args.datasets, args.sentence_level, args.batch_size, args.optimizer_name, args.weight_decay, args.n_epochs, args.embedding_unfreeze, args.learning_rate, args.scheduling_function, args.scheduling_unlock, args.keep_model


if __name__ == '__main__':
    device, seeds, schemes, splits, datasets, sentence_level, batch_size, optimizer_name, weight_decay, n_epochs, embedding_unfreeze, lr, scheduling_function, scheduling_unlock, keep_model = parse_args()
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
    from model_ner import BiLSTM_NER
    from model_trainer import NERTrainer

    torch.device('cuda' if gpu else 'cpu')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    use_cache=False

    seeds = [int(seed) for seed in seeds.split(',')]
    schemes = [str(scheme).upper() for scheme in schemes.split(',')]
    splits = [int(split) for split in splits.split(',')]
    datasets = [str(dataset) for dataset in datasets.split(',')]

    phraser_path = (Path(__file__).parent / '../model/phraser/phraser.pkl').resolve().as_posix()
    vector_path = (Path(__file__).parent / '../model/mat2vec/pretrained_embeddings').resolve().as_posix()

    tokenizer = MaterialsTextTokenizer(phraser_path=phraser_path)

    # corpus parameters
    cased = True
    # cnn parameters
    char_embedding_dim = 25
    char_filter = 5
    char_kernel = 3
    # dropouts
    embedding_dropout_ratio = 0.5
    cnn_dropout_ratio = 0.25
    classifier_dropout_ratio = 0.0
    # attention parameters
    attn_heads = 16
    # lstm parameters
    hidden_dim = 64
    lstm_layers = 2
    lstm_dropout_ratio = 0.1
    attn_dropout_ratio = 0.25
    # parameters for trainer
    max_grad_norm = 1.0

    for seed in seeds:
        for scheme in schemes:
            for split in splits:
                for dataset in datasets:
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)
                    np.random.seed(seed)
                    data_path = (Path(__file__).parent / '../data/{}'.format(dataset)).resolve().as_posix()
                    params = (dataset, 'sentence' if sentence_level else 'paragraph', scheme.lower(), batch_size, optimizer_name, n_epochs, embedding_unfreeze, lr, weight_decay, scheduling_function, scheduling_unlock, seed, split)
                    alias = '{}_{}_{}_crf_{}_{}_{}_{}_{:.0e}_{:.0e}_{}_{}_{}_{}'.format(*params)
                    print('Calculating results for {}'.format(alias))
                    # bilstm paths
                    bilstm_history_path = (Path(__file__).parent / '../model/bilstm/history/{}_history.pt'.format(alias)).resolve().as_posix()
                    bilstm_test_path = (Path(__file__).parent / '../model/bilstm/test/{}_test.pt'.format(alias)).resolve().as_posix()
                    bilstm_model_path = (Path(__file__).parent / '../model/bilstm/{}_model.pt'.format(alias)).resolve().as_posix()

                    data = label_abstracts(format_abstracts(split_abstracts(collect_abstracts(data_path, dataset), (0.1, split/800, split/100), seed), sentence_level), scheme)
                    save_labeled_splits(data_path, dataset, '_{}_{}_{}_{}'.format('sentence' if sentence_level else 'paragraph', scheme.lower(), seed, split), data)
                    corpus = DataCorpus(data_path=data_path, data_name=dataset, alias='_{}_{}_{}_{}'.format('sentence' if sentence_level else 'paragraph', scheme.lower(), seed, split), vector_path=vector_path,
                                        tokenizer=tokenizer, cased=cased, scheme=scheme, batch_size=batch_size, device=device, seed=seed)
                    
                    embedding_dim = corpus.embedding_dim
                    text_vocab_size = len(corpus.text_field.vocab)
                    char_vocab_size = len(corpus.char_field.vocab)
                    tag_vocab_size = len(corpus.label_field.vocab)
                    classes = corpus.classes

                    print(m*'-')
                    print('vocabularies built')
                    print('embedding dimension: {}'.format(embedding_dim))
                    print('unique tokens in text vocabulary: {}'.format(text_vocab_size))
                    print('unique tokens in char vocabulary: {}'.format(char_vocab_size))
                    print('unique tokens in tag vocabulary: {}'.format(tag_vocab_size))
                    print('10 most frequent words in text vocabulary: '+(10*'{} ').format(*corpus.text_field.vocab.freqs.most_common(10)))
                    print('tags: '+(tag_vocab_size*'{} ').format(*classes))
                    print(m*'-')

                    print('train set: {} {}s'.format(len(corpus.train_set), 'sentence' if sentence_level else 'paragraph'))
                    print('valid set: {} {}s'.format(len(corpus.valid_set), 'sentence' if sentence_level else 'paragraph'))
                    print('test set: {} {}s'.format(len(corpus.test_set), 'sentence' if sentence_level else 'paragraph'))
                    print(m*'-')

                    text_pad_idx = corpus.text_pad_idx
                    text_unk_idx = corpus.text_unk_idx
                    char_pad_idx = corpus.char_pad_idx
                    pad_token = corpus.pad_token
                    pretrained_embeddings = corpus.text_field.vocab.vectors

                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)
                    # initialize bilstm
                    bilstm = BiLSTM_NER(input_dim=text_vocab_size, embedding_dim=embedding_dim,
                                        char_input_dim=char_vocab_size, char_embedding_dim=char_embedding_dim,
                                        char_filter=char_filter, char_kernel=char_kernel,
                                        hidden_dim=hidden_dim, output_dim=tag_vocab_size,
                                        lstm_layers=lstm_layers, attn_heads=attn_heads,
                                        embedding_dropout_ratio=embedding_dropout_ratio,
                                        cnn_dropout_ratio=cnn_dropout_ratio, lstm_dropout_ratio=lstm_dropout_ratio,
                                        attn_dropout_ratio=attn_dropout_ratio, classifier_dropout_ratio=classifier_dropout_ratio,
                                        classes=classes, text_pad_idx=text_pad_idx, text_unk_idx=text_unk_idx,
                                        char_pad_idx=char_pad_idx, pad_token=pad_token,
                                        pretrained_embeddings=pretrained_embeddings, scheme=scheme, seed=seed)
                    # print bilstm information
                    print('BiLSTM model initialized with {} trainable parameters'.format(bilstm.count_parameters()))
                    print(bilstm)
                    print(m*'-')

                    # initialize trainer class for bilstm
                    bilstm_trainer = NERTrainer(model=bilstm, device=device)
                    succeeded = True
                    if os.path.exists(bilstm_history_path):
                        print('Already trained {}'.format(alias))
                        history = torch.load(bilstm_history_path)
                        print('{:<10}{:<10}{:10}'.format('epoch', 'training', 'validation'))
                        for i in range(len(history['training'].keys())):
                            metrics = {key: np.mean([batch['micro avg']['f1-score'] for batch in history[key]['epoch_{}'.format(i)]]) for key in ['training', 'validation']}
                            print('{:<10d}{:<10.4f}{:<10.4f}'.format(i, metrics['training'], metrics['validation']))
                    else:
                        try:
                            bilstm_trainer.init_optimizer(optimizer_name, lr, weight_decay)
                            print('training BiLSTM model')
                            print(m*'-')
                            bilstm_trainer.train(n_epoch=n_epochs, train_iter=corpus.train_iter, valid_iter=corpus.valid_iter, embedding_unfreeze=embedding_unfreeze,
                                                 scheduling_function=scheduling_function, scheduling_unlock=scheduling_unlock,
                                                 save_dir=bilstm_model_path, use_cache=use_cache)
                            bilstm_trainer.save_history(history_path=bilstm_history_path)
                            if use_cache:
                                bilstm_trainer.load_state_from_cache('best')
                                bilstm_trainer.save_state(state_path=bilstm_model_path)
                        except:
                            succeeded = False
                            print('Error encountered, skipping')
                    if corpus.train_iter is not None and succeeded:
                        if os.path.exists(bilstm_model_path):
                            # predict test results
                            metrics, test_results = bilstm_trainer.test(test_iter=corpus.test_iter, test_path=bilstm_test_path, state_path=bilstm_model_path)
                        elif os.path.exists(bilstm_test_path):
                            # retrieve test results
                            metrics, test_results = torch.load(bilstm_test_path)
                        # print classification report over test results
                        print(classification_report(test_results['labels'], test_results['predictions'], mode='strict', scheme=bilstm_trainer.metric_scheme))
                        print(m*'-')
                    if not keep_model:
                        try:
                            os.remove(bilstm_model_path)
                        except:
                            print('Saved parameter file {} does not exist'.format(bilstm_model_path))
                    print(m*'-')