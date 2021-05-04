import copy
import numpy as np
from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from seqeval.scheme import IOB1, IOB2, IOBES
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score


class StateCacher(object):
    def __init__(self):
        self.cached = {}


    def store(self, key, state_dict):
        self.cached.update({key: copy.deepcopy(state_dict)})


    def retrieve(self, key):
        if key not in self.cached:
            raise KeyError("target {} was not cached.".format(key))
        return self.cached.get(key)


class NERTrainer(object):
    def __init__(self, model, data, optimizer_cls, criterion_cls, lr, max_grad_norm, device):
        '''
        class for basic functions common to the trainer objects used in this project

        model: the model to be trained
        data: the data corpus to be used for training/validation/testing
        optimizer_cls: the optimizer function (note - pass Adam instead of Adam() or Adam(model.parameters()))
        criterion_cls: the optimization criterion (loss) (note - pass the function name instead of the called function)
        lr: optimizer learning rate
        max_grad_norm: gradient clipping threshold
        device: torch device
        '''
        self.device = device
        # send model to device
        self.model = model.to(self.device)
        self.data = data
        # ignoes the padding in the tags
        self.criterion = criterion_cls(ignore_index=self.data.tag_pad_idx).to(device)
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.optimizer = optimizer_cls(self.model.parameters(), lr=self.lr)
        self.state_cacher = StateCacher()
        self.save_state_to_cache('start')
        # initialize empty lists for training
        self.epoch_metrics = {'training': {}, 'validation': {}}
        self.metric_mode = 'strict'
        if self.data.tag_scheme == 'IOB1':
            self.metric_scheme = IOB1
        elif self.data.tag_scheme == 'IOB2':
            self.metric_scheme = IOB2
        elif self.data.tag_scheme == 'IOBES':
            self.metric_scheme = IOBES
        self.past_epoch = 0
    

    def save_model(self, model_path):
        ''' saves entire model to file '''
        torch.save({'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}, model_path)
    

    def load_model(self, model_path):
        ''' loads entire model from file '''
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model = self.model.to(self.device)


    def save_state_to_cache(self, key):
        self.state_cacher.store('model_{}'.format(key), self.model.state_dict())
        self.state_cacher.store('optimizer_{}'.format(key), self.optimizer.state_dict())
    

    def load_state_from_cache(self, key):
        self.model.load_state_dict(self.state_cacher.retrieve('model_{}'.format(key)))
        self.optimizer.load_state_dict(self.state_cacher.retrieve('optimizer_{}'.format(key)))
        self.model.to(self.device)
    

    def save_history(self, history_path):
        ''' save training histories to file '''
        torch.save(self.epoch_metrics, history_path)
    

    def load_history(self, history_path):
        ''' load training histories from file '''
        self.epoch_metrics = torch.load(history_path)
        self.past_epoch = len(self.epoch_metrics['training'])


    def get_history(self):
        ''' get history '''
        return self.epoch_metrics
    

    def init_scheduler(self, T_max):
        self.lr_scheduler = CosineAnnealingLR(optimizer=self.optimizer, T_max=T_max, verbose=True)
    

    def iterate_batches(self, epoch, n_epoch, iterator, mode):
        '''
        iterates through batchs in an epoch

        epoch: current epoch
        n_epoch: total epochs
        iterator: the iterator to be used for fetching batches
        mode: train, evaluate, or test
        '''
        # initialize lists for batch losses and metrics
        metrics = {'loss': [], 'accuracy_score': [], 'precision_score': [], 'recall_score': [], 'f1_score': []}
        if mode == 'test':
            text_all = []
            char_all = []
            valid_all = []
            prediction_all = []
        # initialize batch range
        batch_range = tqdm(iterator, desc='')
        for batch in batch_range:
            # fetch texts, characters, and tags from batch
            text = batch.text.to(self.device)
            char = batch.char.to(self.device)
            tag = batch.tag.to(self.device)

            # zero out prior gradients for training
            if mode == 'train':
                self.optimizer.zero_grad()

            # output depends on whether conditional random field is used for prediction/loss
            if self.model.use_crf:
                prediction, loss = self.model(text, char, tag)
            else:
                logit = self.model(text, char, tag)
                loss = self.criterion(logit.view(-1, logit.shape[-1]), tag.view(-1))
                logit = logit.detach().cpu().numpy()
                prediction = [list(p) for p in np.argmax(logit, axis=2)]

            # send the true tags to python list on the cpu
            true = list(tag.to('cpu').numpy())

            # put the prediction tags and valid tags into a nested list form for the scoring metrics
            prediction_tags = [[self.data.tag_field.vocab.itos[ii] if ii != self.data.tag_pad_idx else 'O' for ii, jj in zip(i, j) if self.data.tag_field.vocab.itos[jj] != self.data.pad_token] for i, j in zip(prediction, true)]
            valid_tags = [[self.data.tag_field.vocab.itos[ii] for ii in i if self.data.tag_field.vocab.itos[ii] != self.data.pad_token] for i in true]

            if mode == 'test':
                text_all.extend(list(text.cpu().numpy()))
                char_all.extend(list(char.cpu().numpy()))
                valid_all.extend(label_tags)
                prediction_all.extend(prediction_tags)

            # calculate the accuracy and f1 scores
            accuracy = accuracy_score(valid_tags, prediction_tags)
            precision = precision_score(valid_tags, prediction_tags, mode=self.metric_mode, scheme=self.metric_scheme)
            recall = recall_score(valid_tags, prediction_tags, mode=self.metric_mode, scheme=self.metric_scheme)
            f1 = f1_score(valid_tags, prediction_tags, mode=self.metric_mode, scheme=self.metric_scheme)

            # append to the lists
            metrics['loss'].append(loss.item())
            metrics['accuracy_score'].append(accuracy)
            metrics['precision_score'].append(precision)
            metrics['recall_score'].append(recall)
            metrics['f1_score'].append(f1)

            # backpropagate the gradients and step the optimizer forward
            if mode == 'train':
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()

            # calculate means across the batches so far
            means = [np.mean(metrics[metric]) for metric in metrics.keys()]
            # display progress
            batch_range.set_description('| epoch: {:d}/{:d} | {} | loss: {:.4f} | accuracy: {:.4f} | precision: {:.4f} | recall: {:.4f} | f1: {:.4f} |'.format(self.past_epoch+epoch+1, self.past_epoch+n_epoch, mode, *means))
        # return the batch losses and metrics
        if mode == 'test':
            return metrics, text_all, char_all, valid_all, prediction_all
        else:
            return metrics
    

    def train_evaluate_epoch(self, epoch, n_epoch, iterator, mode):
        '''
        train or evaluate epoch (calls the iterate_batches method from a subclass that inherits from this class)

        epoch: current epoch
        n_epoch: total epochs
        iterator: the iterator to be used for fetching batches
        mode: train, evaluate, or test
        '''
        if mode == 'train':
            # make sure the model is set to train if it is training
            self.model.train()
            # train all of the batches and collect the batch/epoch loss/metrics
            metrics = self.iterate_batches(epoch, n_epoch, iterator, mode)
        elif mode == 'validate':
            # make sure the model is set to evaluate if it is evaluating
            self.model.eval()
            # turn off gradients for evaluating
            with torch.no_grad():
                # evaluate all of the batches and collect the batch/epoch loss/metrics
                metrics = self.iterate_batches(epoch, n_epoch, iterator, mode)
                if (epoch+1) >= int(np.floor(0.72*n_epoch)) and (epoch+1) < n_epoch:
                    self.lr_scheduler.step()
        elif mode == 'test':
            # make sure the model is set to evaluate if it is evaluating
            self.model.eval()
            # turn off gradients for evaluating
            with torch.no_grad():
                # evaluate all of the batches and collect the batch/epoch loss/metrics
                metrics, text, char, valid, prediction = self.iterate_batches(epoch, n_epoch, iterator, mode)
        # return batch/epoch loss/metrics
        if mode == 'test':
            return metrics, text, char, valid, prediction
        else:
            return metrics
    

    def train(self, n_epoch):
        '''
        trains the model (with validation)

        n_epoch: number of training epochs
        '''
        best_validation_f1 = 0.0
        self.init_scheduler(T_max=int(np.ceil(0.28*n_epoch)))
        for epoch in range(n_epoch):
            # training
            train_metrics = self.train_evaluate_epoch(epoch, n_epoch, self.data.train_iter, 'train')
            # validation
            valid_metrics = self.train_evaluate_epoch(epoch, n_epoch, self.data.valid_iter, 'validate')
            # save best
            validation_f1 = np.mean(valid_metrics['f1_score'])
            if validation_f1 > best_validation_f1:
                best_validation_f1 = validation_f1
                self.save_state_to_cache('best_validation_f1')
            # append histories
            self.epoch_metrics['training']['epoch_{}'.format(epoch)] = train_metrics
            self.epoch_metrics['validation']['epoch_{}'.format(epoch)] = valid_metrics
    
    
    def test(self, test_path):
        ''' evaluates the test set '''
        metrics, text, char, valid, prediction = self.train_evaluate_epoch(0, 1, self.data.test_iter, 'test')
        torch.save((metrics, text, char, valid, prediction), test_path)
        return metrics, text, char, valid, prediction
