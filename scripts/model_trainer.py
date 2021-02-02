import copy
import numpy as np
from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from seqeval.metrics import accuracy_score, f1_score


class StateCacher(object):
    def __init__(self):
        self.cached = {}


    def store(self, key, state_dict):
        self.cached.update({key: copy.deepcopy(state_dict)})


    def retrieve(self, key):
        if key not in self.cached:
            raise KeyError("target {} was not cached.".format(key))
        return self.cached.get(key)


class ModelTrainer(object):
    def __init__(self, model, data, optimizer_cls, criterion_cls, full_finetuning, max_grad_norm, device):
        '''
        
        class for basic functions common to the trainer objects used in this project

        model: the model to be trained
        data: the data corpus to be used for training/validation/testing
        optimizer_cls: the optimizer function (note - pass Adam instead of Adam() or Adam(model.parameters()))
        criterion_cls: the optimization criterion (loss) (note - pass the function name instead of the called function)
        device: torch device

        '''
        self.device = device
        # send model to device
        self.model = model.to(self.device)
        self.data = data
        # ignoes the padding in the tags
        self.criterion = criterion_cls(ignore_index=self.data.tag_pad_idx).to(device)
        self.max_grad_norm = max_grad_norm
        self.full_finetuning = full_finetuning
        if self.model.bert_model:
            self.finetuning()
            self.optimizer = optimizer_cls(self.optimizer_grouped_parameters, lr=3e-5, eps=1e-8)
        else:
            self.optimizer = optimizer_cls(self.model.parameters())
        self.state_cacher = StateCacher()
        self.save_state_to_cache('start')
        self.lr_scheduler = None
        # initialize empty lists for training
        self.epoch_metrics = {'training': {}, 'validation': {}}
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
        self.past_epoch = len(self.metrics['train'])


    def get_history(self):
        ''' get history '''
        return self.epoch_metrics
    

    def finetuning(self):
        ''' determines the parameters to be finetuned '''
        if self.model.bert_model:
            # optimize all network parameters if doing a full finetuning
            if self.full_finetuning:
                param_optimizer = list(self.model.named_parameters())
                no_decay = ['bias', 'gamma', 'beta']
                self.optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
                                                    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],'weight_decay_rate': 0.0}]
            # otherwise, optimize only the classification layers
            else:
                param_optimizer = list(self.model.bert.classifier.named_parameters())
                if self.model.use_crf:
                    param_optimizer += list(self.model.crf.named_parameters())
                self.optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
    

    def schedule_lr(self):
        self.lr_scheduler = ReduceLROnPlateau(optimizer=self.optimizer, patience=8, factor=0.25, mode='max', verbose=True)
    

    def train_evaluate_epoch(self, epoch, n_epoch, iterator, train, mode):
        '''
        
        train or evaluate epoch (calls the iterate_batches method from a subclass that inherits from this class)

        epoch: current epoch
        n_epoch: total epochs
        iterator: the iterator to be used for fetching batches
        train: switch for whether or not to train this epoch
        mode: string that just labels the epoch in the output

        '''
        if train:
            # make sure the model is set to train if it is training
            self.model.train()
            # train all of the batches and collect the batch/epoch loss/metrics
            metrics = self.iterate_batches(epoch, n_epoch, iterator, train, mode)
        else:
            # make sure the model is set to evaluate if it is evaluating
            self.model.eval()
            # turn off gradients for evaluating
            with torch.no_grad():
                # evaluate all of the batches and collect the batch/epoch loss/metrics
                metrics = self.iterate_batches(epoch, n_epoch, iterator, train, mode)
                if self.lr_scheduler:
                    self.lr_scheduler.step(np.mean(batch_f1))
        # return batch/epoch loss/metrics
        return metrics
    

    def train(self, n_epoch):
        '''

        trains the model (with validation)

        n_epoch: number of training epochs

        '''
        for epoch in range(n_epoch):
            # training
            train_metrics = self.train_evaluate_epoch(epoch, n_epoch, self.data.train_iter, True, 'train')
            # validation
            valid_metrics = self.train_evaluate_epoch(epoch, n_epoch, self.data.valid_iter, False, 'validate')
            # append histories
            self.epoch_metrics['training']['epoch_{}'.format(epoch)] = train_metrics
            self.epoch_metrics['validation']['epoch_{}'.format(epoch)] = valid_metrics
    
    
    def test(self):
        ''' evaluates the test set '''
        return self.train_evaluate_epoch(0, 1, self.data.test_iter, False, 'test')


class NERTrainer(ModelTrainer):
    def __init__(self, model, data, optimizer_cls, criterion_cls, full_finetuning, max_grad_norm, device):
        ''' trainer for named entity recognition model. inherits from model trainer class '''
        super().__init__(model, data, optimizer_cls, criterion_cls, full_finetuning, max_grad_norm, device)
    

    def iterate_batches(self, epoch, n_epoch, iterator, train, mode):
        '''
        
        iterates through batchs in an epoch

        epoch: current epoch
        n_epoch: total epochs
        iterator: the iterator to be used for fetching batches
        train: switch for whether or not to train this epoch
        mode: string that just labels the epoch in the output

        '''
        # initialize lists for batch losses and metrics
        metrics = {'loss': [], 'accuracy_score': [], 'f1_score': []}
        # initialize mode for metrics
        if self.data.format == 'IOB2':
            metric_mode = None
        elif self.data.format == 'BILOU' or self.data.format == 'BIOES':
            metric_mode = 'strict'
        # initialize batch range
        batch_range = tqdm(iterator, desc='')
        for batch in batch_range:
            # fetch texts, characters, and tags from batch
            if self.model.bert_model:
                text = batch.text.to(self.device)
                tag = batch.text.to(self.device)
                # ignore pad tokens in the attention mask
                attention_mask = torch.tensor(np.array([[tt != self.data.text_field.pad_token for tt in t] for t in text])).to(self.device)
            else:
                text = batch.text.to(self.device)
                char = batch.char.to(self.device)
                tag = batch.tag.to(self.device)

            # zero out prior gradients for training
            if train:
                self.optimizer.zero_grad()

            # output depends on whether conditional random field is used for prediction/loss
            if self.model.bert_model:
                if self.model.use_crf:
                    prediction, loss = self.model(text, attention_mask, tag)
                else:
                    logit = self.model(text, attention_mask, tag)
                    loss = self.criterion(logit.view(-1, logit.shape[-1]), tag.view(-1))
                    logit = logit.detach().cpu().numpy()
                    prediction = [list(p) for p in np.argmax(logit, axis=2)]
            else:
                if self.model.use_crf:
                    prediction, loss = self.model(text, char, tag)
                else:
                    logit = self.model(text, char, tag)
                    loss = self.criterion(logit.view(-1, logit.shape[-1]), tag.view(-1))
                    logit = logit.detach().cpu().numpy()
                    prediction = [list(p) for p in np.argmax(logit, axis=2)]

            # send the true tags to python list on the cpu
            if self.model.bert_model:
                # remove first token id in each sentence (to make crf mask work)
                true = list(tag.to('cpu').numpy()[:, 1:])
            else:
                true = list(tag.to('cpu').numpy())

            # put the prediction tags and valid tags into a nested list form for the scoring metrics
            prediction_tags = [[self.data.tag_field.vocab.itos[ii] for ii, jj in zip(i, j) if self.data.tag_field.vocab.itos[jj] != self.data.pad_token] for i, j in zip(prediction, true)]
            valid_tags = [[self.data.tag_field.vocab.itos[ii] for ii in i if self.data.tag_field.vocab.itos[ii] != self.data.pad_token] for i in true]

            # calculate the accuracy and f1 scores
            accuracy = accuracy_score(valid_tags, prediction_tags)
            f1 = f1_score(valid_tags, prediction_tags, mode=metric_mode)

            # append to the lists
            metrics['loss'].append(loss.item())
            metrics['accuracy_score'].append(accuracy)
            metrics['f1_score'].append(f1)

            # backpropagate the gradients and step the optimizer forward
            if train:
                loss.backward()
                if self.model.bert_model:
                    torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()

            # calculate means across the batches so far
            means = [np.mean(metrics[metric]) for metric in ['loss', 'accuracy_score', 'f1_score']]
            # display progress
            batch_range.set_description('| epoch: {:d}/{:d} | {} | loss: {:.4f} | accuracy_score: {:.4f} | f1_score: {:.4f} |'.format(self.past_epoch+epoch+1, self.past_epoch+n_epoch, mode, *means))
        # return the batch losses and metrics
        return metrics