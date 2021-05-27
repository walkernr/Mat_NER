import copy
import numpy as np
from tqdm import tqdm
import torch
from torchtools.optim import AdamW, RangerLars
from torch.optim.lr_scheduler import LambdaLR
from seqeval.scheme import IOB1, IOB2, IOBES
from seqeval.metrics import accuracy_score, classification_report


class StateCacher(object):
    '''
    An object that caches the state of a training model (parameters and optimizer)
    '''
    def __init__(self):
        '''
        Initializes the stach cacher object
            Arguments:
                None
            Returns:
                None
        '''
        # initialize empty dictionary of cached states
        self.cached = {}


    def store(self, key, state_dict):
        '''
        Caches a state
            Arguments:
                key: Identifier for state
                state_dict: Dictionary of states
            Returns:
                None
        '''
        # update dictionary of states
        self.cached.update({key: copy.deepcopy(state_dict)})


    def retrieve(self, key):
        '''
        Retrieves a cached state
            Arguments:
                key: Identifier for state
            Returns:
                State dictionary
        '''
        # raise error if key does not exist
        if key not in self.cached:
            raise KeyError('target {} was not cached.'.format(key))
        # return state dictionary
        return self.cached.get(key)


class NERTrainer(object):
    def __init__(self, model, device):
        '''
        class for basic functions common to the trainer objects used in this project

        model: the model to be trained
        data: the data corpus to be used for training/validation/testing
        lr: optimizer learning rate
        max_grad_norm: gradient clipping threshold
        device: torch device
        '''
        self.device = device
        # send model to device
        self.model = model.to(self.device)
        # ignoes the padding in the tags
        self.max_grad_norm = 1.0
        self.state_cacher = StateCacher()
        self.metric_mode = 'strict'
        if self.model.scheme == 'IOB1':
            self.metric_scheme = IOB1
        elif self.model.scheme == 'IOB2':
            self.metric_scheme = IOB2
        elif self.model.scheme == 'IOBES':
            self.metric_scheme = IOBES
        self.past_epoch = 0
    

    def save_state(self, state_path, optimizer=True):
        '''
        Saves the state of the model and optimizer to file
            Arguments:
                state_path: Path to save the state to
                optimizer: Boolean controlling whether to save the optimizer state
            Returns:
                None
        '''
        # state consists of classes and model parameter state dictionary
        state = {'classes': self.model.classes,
                 'model_state_dict': self.model.state_dict()}
        # if optimizer, include state dictionary
        if optimizer:
            state['optimizer_state_dict'] = self.optimizer.state_dict()
        # save to path
        torch.save(state, state_path)
    

    def load_state(self, state_path, optimizer=True):
        '''
        Loads the state of the model and optimizer from file
            Arguments:
                state_path: Path to load the state from
                optimizer: Boolean controlling whether to save the optimizer state
            Returns:
                None
        '''
        # load checkpoint and map to device
        checkpoint = torch.load(state_path, map_location=torch.device(self.device))
        # set classes in model
        self.model.classes = checkpoint['classes']
        # rebuild model layers
        self.model.build_model()
        # send model to device
        self.model.to(self.device)
        # load model parameters from state dictionary
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # if optimizer, load state
        if optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    def save_state_to_cache(self, key, optimizer=True):
        '''
        Saves state to cache
            Arguments:
                key: Identification key
                optimizer: Boolean controlling whether to save the optimizer state
            Returns:
                None
        '''
        # cache model state dictionary with key
        self.state_cacher.store('model_state_dict_{}'.format(key), self.model.state_dict())
        # if optimizer, cache optimizer state dictionary with key
        if optimizer:
            self.state_cacher.store('optimizer_state_dict_{}'.format(key), self.optimizer.state_dict())
    

    def load_state_from_cache(self, key, optimizer=True):
        '''
        Loads state from cache
            Arguments:
                key: Identification key
                optimizer: Boolean controlling whether to save the optimizer state
            Returns:
                None
        '''
        # load model parameters from state dictionary with key
        self.model.load_state_dict(self.state_cacher.retrieve('model_state_dict_{}'.format(key)))
        # if optimizer, load optimzier state dictionary with key
        if optimizer:
            self.optimizer.load_state_dict(self.state_cacher.retrieve('optimizer_state_dict_{}'.format(key)))
    

    def save_history(self, history_path):
        ''' 
        Save training histories to file
            Arguments:
                history_path: Path to save the history to
            Returns:
                None
        '''
        # save epoch metrics
        torch.save(self.epoch_metrics, history_path)
    

    def load_history(self, history_path):
        '''
        Load training histories from file
            Arguments:
                history_path: Path to load the history from
            Returns:
                None
        '''
        # load epoch metrics from path
        self.epoch_metrics = torch.load(history_path)
        # set past epochs
        self.past_epoch = len(self.epoch_metrics['training'].keys())


    def return_history(self):
        '''
        Return history
            Arguments:
                None
            Returns:
                Dictionary of epoch metrics
        '''
        # return epoch metrics
        return self.epoch_metrics

    
    def init_optimizer(self, optimizer_name, lr):
        '''
        Initialize optimizer
            Arguments:
                optimizer_name: Name of optimizer (adamw or rangerlars)
                lr: Learning rate
            Returns:
                None
        ''' 
        # optimizer dict
        optimizers = {'adamw': AdamW, 'rangerlars': RangerLars}
        # default to AdamW if invalid optimizer name provided
        if optimizer_name not in optimizers.keys():
            optimizer_name = 'adamw'
            print('Reverted to default optimizer (AdamW)')
        # construct optimizer
        self.optimizer = optimizers[optimizer_name]([{'params': self.model.parameters(), 'lr': lr}])
    

    def init_scheduler(self, n_epoch, function_name='exponential'):
        '''
        Initializes learning rate scheduler
            Arguments:
                n_epoch: Number of training epochs
                function_name: Name of scheduling funtion (linear, exponential, or cosine)
            Returns:
                None
        '''
        # dictionary of functions
        functions = {'linear': lambda epoch: (n_epoch-epoch)/(n_epoch),
                     'exponential': lambda epoch: 0.01**(epoch/(n_epoch-1)),
                     'cosine': lambda epoch: 0.5*(1+np.cos(epoch/n_epoch)*np.pi)}
        # default to linear if invalid function name provided
        if function_name not in functions.keys():
            function_name = 'linear'
            print('Reverted to default scheduling function (linear)')
        # construct scheduler
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=functions[function_name], verbose=True)
    

    def process_labels(self, label_ids, prediction_ids, attention_mask):
        labels = []
        predictions = []
        for i in range(len(attention_mask)):
            l = []
            p = []
            k = 0
            for j in range(len(attention_mask[i])):
                if attention_mask[i][j] == 1:
                    l.append(self.model.classes[label_ids[i][j]])
                    p.append(self.model.classes[prediction_ids[i][k]])
                    k += 1
            labels.append(l)
            predictions.append(p)
        return {'labels': labels, 'predictions': predictions}
    
    
    def process_ids(self, token_ids, prediction_ids):
        '''
        Processes ids into tokens and labels
            Arguments:
                token_ids: Sequence token ids
                prediction_ids: Sequence prediction ids
            Returns:
                Dictionary of text and annotations by word, sentence, paragraph e.g. [[[{'text': text, 'annotation': annotation},...],...],...]
        '''
        # for entry index
        for i in range(len(token_ids)):
            # identify padding indices
            idx = np.where(token_ids[i] == self.model.text_pad_idx)
            # trim padding indices
            token_ids[i] = np.delete(token_ids[i], idx, axis=0)
            prediction_ids[i] = np.delete(prediction_ids[i], idx, axis=0)
        # convert token ids to tokens for each entry
        toks = [self.model.tokenizer.convert_ids_to_tokens(sequence) for sequence in token_ids]
        # convert label ids to classifications for each entry
        lbls = [[self.model.classes[l] for l in sequence] for sequence in prediction_ids]
        # initialize empty lists of entries for tokens and labels
        # ctoks for c(ombined)tok(en)s, referencing that we will be merging the tokens split by the BERT tokenizer
        # slbls for s(ingle)l(a)b(e)ls, referencing that we often only have a single label for multiple subtokens
        ctoks = []
        slbls = []
        # for index in entries
        for i in range(len(valid_mask)):
            # initialize a list of an empty list, each sublist will be a sentence
            ctok = [[]]
            slbl = [[]]
            # initialize special subindices
            # sentence index (only stepped forward on new sentence)
            ii = 0
            # index within sentence (only stepped forward within sentence and reset on new sentence)
            jj = 0
            # index in labels (only stepped forward on valid indices)
            kk = 0
            # for index in sequence
            for j in range(len(valid_mask[i])):
                # if the index is valid
                if valid_mask[i][j] == 1:
                    # if the index is a cls token, pass
                    if toks[i][j] == self.cls_dict['token']:
                        pass
                    # if the index is a sep token
                    elif toks[i][j] == self.sep_dict['token']:
                        # if the current index is not the last index in the sequence
                        if j < len(valid_mask[i])-1:
                            # append new empty sentence
                            ctok.append([])
                            slbl.append([])
                            # step sentence index forward
                            ii += 1
                            # reset index within sentence to zero
                            jj = 0
                    # if valid token and not special token
                    else:
                        # append token to combined tokens
                        ctok[ii].append(toks[i][j])
                        # append label to single labels
                        slbl[ii].append(lbls[i][kk] if lbls[i][kk] == 'O' else lbls[i][kk].split('-')[1])
                        # step index within sentence forward
                        jj += 1
                    # step index within labels
                    kk += 1
                # if not valid index
                else:
                    # if token is suffix (as determined by BERT tokenizer with ##)
                    if '##' in toks[i][j]:
                        # combine with prior token while filtering out the suffix indication (##)
                        ctok[ii][jj-1] += toks[i][j].replace('##', '')
                    # otherwise
                    else:
                        # combine with prior token
                        ctok[ii][jj-1] += toks[i][j]
            # append sentences
            ctoks.append(ctok)
            slbls.append(slbl)
        # empty intity list
        entities = []
        # class types
        class_types = sorted(list(set([class_.split('-')[1] for class_ in self.model.classes if class_ != 'O'])))
        # for entry
        for i in range(len(slbls)):
            # initialize empty dictionary of extracted entities
            entry_entities = {class_type: set([]) for class_type in class_types}
            # for sentence in entry
            for j in range (len(slbls[i])):
                # extract beginning indices for entities
                entity_idx = [k for k in range(len(slbls[i][j])) if slbls[i][j][k] != slbls[i][j][k-1]]
                # append ending sequence
                entity_idx.append(len(slbls[i][j]))
                # for entity beginning index
                for k in range(len(entity_idx)):
                    # if not last index (last index in sequence)
                    if k != len(entity_idx)-1:
                        # if entity is not outside
                        if slbls[i][j][entity_idx[k]] != 'O':
                            # append joined entity to dictionary
                            entry_entities[slbls[i][j][entity_idx[k]]].add(' '.join(ctoks[i][j][entity_idx[k]:entity_idx[k+1]]))
            # append entry entity dictionary
            entities.append(entry_entities)
        # construct dictionary of (text, annotation) pairs under tokens and a dictionary of entities under entities
        annotations = [{'tokens': [[{'text': t, 'annotation': l} for t, l in zip(ctoks_sequence, slbls_sequence)]
                                   for ctoks_sequence, slbls_sequence in zip(ctoks_entry, slbls_entry)],
                        'entities': entities_entry}
                       for ctoks_entry, slbls_entry, entities_entry in zip(ctoks, slbls, entities)]
        return annotations


    def iterate_batches(self, epoch, n_epoch, iterator, mode):
        '''
        Iterates through batches in epoch
            Arguments:
                epoch: Current epoch
                n_epoch: Total number of epochs
                iterator: Dataloader
                mode: Model mode e.g. train, evaluate, test, or predict
            Returns:
                mode == train or validate: metrics
                mode == test: metrics and test results
                mode == predict: prediction results
        '''
        # if mode is not prediction, initialize empty list for metrics
        if mode != 'predict':
            metrics = []
        # if mode is test, initialize dictionary of labels and predictions
        if mode == 'test':
            test_results = {'labels': [], 'predictions': []}
        # if mode is predict, initialize dictionary of token_ids, and prediction_ids
        if mode == 'predict':
            prediction_results = {'token_ids': [], 'prediction_ids': []}
        # initialize batch range
        batch_range = tqdm(iterator, desc='')
        for batch in batch_range:
            # collect inputs from batch
            inputs = {'token_ids': batch.text.to(self.device),
                      'char_ids': batch.char.to(self.device)}
            if mode != 'predict':
                    inputs['label_ids'] = batch.label.to(self.device)
            inputs['attention_mask'] = (inputs['token_ids'] != self.model.text_pad_idx)

            # zero out prior gradients for training
            if mode == 'train':
                self.optimizer.zero_grad()

            # if mode is not predict, collect loss and prediction ids and then process labels
            if mode != 'predict':
                loss, prediction_ids = self.model.forward(**inputs)
                batch_results = self.process_labels(list(inputs['label_ids'].cpu().numpy()), prediction_ids, list(inputs['attention_mask'].cpu().numpy()))
            # if mode is predict, only collect prediction ids
            else:
                prediction_ids = self.model.forward(**inputs)
            
            # if mode is test, extend the list of batch results in the test results dictionary for the labels and predictions keys
            if mode == 'test':
                for key in test_results.keys():
                    test_results[key].extend(batch_results[key])
            # if mode is predict, extend lists of token_ids, attention_masks, valid_masks, and prediction_ids by keys
            if mode == 'predict':
                prediction_results['prediction_ids'].extend(prediction_ids)
                prediction_results['token_ids'].extend(list(inputs['token_ids'].cpu().numpy()))

            # if mode is not predice
            if mode != 'predict':
                # generate classification report from labels and predictions (with filtered out special tokens)
                report = classification_report(batch_results['labels'], batch_results['predictions'], mode=self.metric_mode, scheme=self.metric_scheme, output_dict=True)
                # add accuracy score to report
                report['accuracy'] = accuracy_score(batch_results['labels'], batch_results['predictions'])
                # add loss to report
                report['loss'] = loss.item()
                # append report to metrics
                metrics.append(report)
                # calculate rolling means for precision, recall and f1-score
                means = {m: np.mean([r['micro avg'][m] for r in metrics]) for m in ['precision', 'recall', 'f1-score']}
                # calculate rolling means for accuracy and loss
                means['accuracy'] = np.mean([r['accuracy'] for r in metrics])
                means['loss'] = np.mean([r['loss'] for r in metrics])

            # backpropagate the gradients and step the optimizer forward
            if mode == 'train':
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()

            # if mode is not predict
            if mode != 'predict':
                # display epoch progress, mode, and rolling averages alongside batch progress
                msg = '| epoch: {:d}/{:d} | {} | loss: {:.4f} | accuracy: {:.4f} | precision: {:.4f} | recall: {:.4f} | f1-score: {:.4f} |'
                info = (self.past_epoch+epoch+1, self.past_epoch+n_epoch, mode, means['loss'], means['accuracy'], means['precision'], means['recall'], means['f1-score'])
            # if mode is predict
            else:
                # display epoch progress, mode, and batch progress
                msg = '| epoch: {:d}/{:d} | {} |'
                info = (self.past_epoch+epoch+1, self.past_epoch+n_epoch, mode)
            # set progress bar description
            batch_range.set_description(msg.format(*info))
        # return statements
        if mode == 'test':
            return metrics, test_results
        elif mode != 'predict':
            return metrics
        else:
            return prediction_results
    

    def train_evaluate_epoch(self, epoch, n_epoch, iterator, mode):
        '''
        Trains or evaluate the model for an epoch
            Arguments:
                epoch: Current epoch
                n_epoch: Total number of epochs
                iterator: Dataloader
                mode: Model mode e.g. train, evaluate, test, or predict
            Returns:
                mode == train or validate: metrics
                mode == test: metrics and test results
                mode == predict: prediction results
        '''
        # if training
        if mode == 'train':
            # make sure the model is set to train
            self.model.train()
            # train all of the batches and collect the metrics
            metrics = self.iterate_batches(epoch, n_epoch, iterator, mode)
        # if validating
        elif mode == 'valid':
            # make sure the model is set to evaluate
            self.model.eval()
            # turn off gradients for evaluating
            with torch.no_grad():
                # evaluate all of the batches and collect the metrics
                metrics = self.iterate_batches(epoch, n_epoch, iterator, mode)
        # if testing
        elif mode == 'test':
            # make sure the model is set to evaluate
            self.model.eval()
            # turn off gradients for evaluating
            with torch.no_grad():
                # evaluate all of the batches and collect the metrics
                metrics, test_results = self.iterate_batches(epoch, n_epoch, iterator, mode)
        # if predicting
        elif mode == 'predict':
            # make sure the model is set to evaluate
            self.model.eval()
            # turn off gradients for evaluating
            with torch.no_grad():
                # evaluate all of the batches and collect the prediction results
                prediction_results = self.iterate_batches(epoch, n_epoch, iterator, mode)
        # return statements
        if mode == 'test':
            return metrics, test_results
        elif mode != 'predict':
            return metrics
        else:
            return prediction_results
    

    def train(self, n_epoch, train_iter, valid_iter, scheduling_function, save_dir, use_cache):
        '''
        Trains the model with validation if a validation iterator is provided
            Arguments:
                n_epoch: Total number of epochs
                train_iter: Training dataloader
                valid_iter: Validation dataloader
                save_dir: Save directory for model and optimizer state
                use_cache: Boolean that controls whether to use the cache for saving the model/optimizer state. If False, states are saved to disk at save_dir
            Returns:
                None
        '''
        # initialize dictionary of epoch metrics
        self.epoch_metrics = {'training': {}}
        if valid_iter is not None:
            self.epoch_metrics['validation'] = {}
        
        self.init_scheduler(n_epoch, scheduling_function)

        # freeze mat2vec embeddings
        for param in self.model.embedding.parameters():
            param.requires_grad = False
        print('Mat2Vec embeddings frozen')
        # initialize best validation f1
        best_validation_f1 = 0.0

        # for each epoch
        for epoch in range(n_epoch):
            # training
            train_metrics = self.train_evaluate_epoch(epoch, n_epoch, train_iter, 'train')
            # append history
            self.epoch_metrics['training']['epoch_{}'.format(self.past_epoch+epoch)] = train_metrics
            if valid_iter:
                # validation
                valid_metrics = self.train_evaluate_epoch(epoch, n_epoch, valid_iter, 'valid')
                # append_history
                self.epoch_metrics['validation']['epoch_{}'.format(self.past_epoch+epoch)] = valid_metrics
                # save best
                validation_f1 = np.mean([batch_metrics['micro avg']['f1-score'] for batch_metrics in valid_metrics])
                if validation_f1 >= best_validation_f1:
                    best_validation_f1 = validation_f1
                    if use_cache:
                        self.save_state_to_cache('best')
                    else:
                        self.save_state(save_dir)
            # step scheduler forward
            if epoch < n_epoch-1:
                self.scheduler.step()
    
    
    def test(self, test_iter, test_path, state_path=None):
        '''
        Evaluates the tests set and saves the predictions alongside the ground truths
            Arguments:
                test_iter: Test dataloader
                test_path: Path to save the test results to
                state_path: Path to load the model state from
            Returns:
                metrics and test results
        '''
        # if state path provided, load state (excluding optimizer)
        if state_path is not None:
            self.load_state(state_path, optimizer=False)
        # evaluate the test set
        metrics, test_results = self.train_evaluate_epoch(0, 1, test_iter, 'test')
        # save the test metrics and results
        torch.save((metrics, test_results), test_path)
        # return the test metrics and results
        return metrics, test_results
    

    def predict(self, predict_iter, predict_path, state_path=None):
        '''
        Predicts classifications for a dataset
            Arguments:
                predict_iter: Prediction dataloader
                predict_path: Path to save the predictions to
                state_path: Path to load the model state from
            Returns:
                Dictionary of text and annotations by word, sentence, paragraph e.g. [[[{'text': text, 'annotation': annotation},...],...],...]
        '''
        # if state path provided, load state (excluding optimizer)
        if state_path is not None:
            self.load_state(state_path, optimizer=False)
        # evaluate the prediction set
        prediction_results = self.train_evaluate_epoch(0, 1, predict_iter, 'predict')
        # process the predictions into annotations
        annotations = self.process_ids(prediction_results['token_ids'], prediction_results['prediction_ids'])
        # save annotations
        torch.save(annotations, predict_path)
        # return the annotations
        return annotations
