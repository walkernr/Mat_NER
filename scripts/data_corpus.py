import numpy as np
from collections import Counter
import gensim
import torch
from torchtext.data import Field, NestedField, Iterator
from torchtext.datasets import SequenceTaggingDataset
from torchtext.vocab import Vocab


class DataCorpus(object):
    def __init__(self, data_path, data_name, alias, vector_path, tokenizer, cased, scheme, batch_size, device, seed):
        self.data_path = data_path
        self.data_name = data_name
        self.alias = alias
        self.vector_path = vector_path
        self.tokenizer = tokenizer
        self.cased = cased
        self.scheme = scheme
        self.batch_size = batch_size
        self.device = device
        self.seed = seed
        self.pad_token = self.tokenizer.pad_token
        self.unk_token = self.tokenizer.unk_token
        self.label_pad_token = 'O'
        self.initialize_fields()
        self.load_data()
        self.build_label_vocabulary()
        self.build_text_vocabulary()
        self.build_char_vocabulary()
        self.initialize_iterators()


    def insert_initial_pad_label(self, label):
        label.insert(0, self.pad_token)
        return label


    def initialize_fields(self):
        self.text_field = Field(tokenize=self.tokenizer.tokenize, preprocessing=self.tokenizer.process, lower=not self.cased,
                                pad_token=self.pad_token, unk_token=self.unk_token,
                                batch_first=True)
        self.label_field = Field(pad_token=self.label_pad_token, unk_token=None,
                                 batch_first=True)
        char_nesting_field = Field(tokenize=list, pad_token=self.pad_token, batch_first=True)
        self.char_field = NestedField(char_nesting_field)

    
    def load_data(self):
        fields = ((('text', 'char'), (self.text_field, self.char_field)),
                  ('label', self.label_field))
        if self.seed:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            np.random.seed(self.seed)
        self.train_set, self.valid_set, self.test_set = SequenceTaggingDataset.splits(fields=fields, path=self.data_path+'/split/',
                                                                                      train=self.data_name+self.alias+'_train.tsv',
                                                                                      validation=self.data_name+self.alias+'_valid.tsv',
                                                                                      test=self.data_name+self.alias+'_test.tsv')
        full_labels = np.concatenate([np.loadtxt(self.data_path+'/split/'+self.data_name+self.alias+'_{}.tsv'.format(split), delimiter='\t', dtype=str)[:, 1] for split in ('train', 'valid', 'test')])
        labels = np.unique([label.split('-')[1] for label in full_labels if '-' in label])
        if self.scheme in ('IOB1', 'IOB2'):
            prefixes = ['I', 'B']
        elif self.scheme == 'IOBES':
            prefixes = ['I', 'B', 'E', 'S']
        self.classes = [['{}-{}'.format(prefix, label) for prefix in prefixes] for label in labels]


    def build_label_vocabulary(self):
        self.label_field.build_vocab(self.classes)
        # self.label_pad_idx = self.label_field.vocab.stoi[self.pad_token]
        self.classes = self.label_field.vocab.itos


    def build_text_vocabulary(self):
        self.vector_model = gensim.models.word2vec.Word2Vec.load(self.vector_path)
        self.embedding_dim = self.vector_model.vector_size
        word_freq = {word: self.vector_model.wv.vocab[word].count for word in self.vector_model.wv.vocab}
        word_counter = Counter(word_freq)
        self.text_field.vocab = Vocab(word_counter)
        vectors = []
        for word, idx in self.text_field.vocab.stoi.items():
            if word in self.vector_model.wv.vocab.keys():
                vectors.append(torch.as_tensor(self.vector_model.wv[word].tolist()))
            else:
                vectors.append(torch.zeros(self.embedding_dim))
        self.text_field.vocab.set_vectors(stoi=self.text_field.vocab.stoi, vectors=vectors, dim=self.embedding_dim)
        self.text_pad_idx = self.text_field.vocab.stoi[self.text_field.pad_token]
        self.text_unk_idx = self.text_field.vocab.stoi[self.text_field.unk_token]

    
    def build_char_vocabulary(self):
        self.char_field.build_vocab(self.train_set.char)
        self.char_pad_idx = self.char_field.vocab.stoi[self.pad_token]
    

    def initialize_iterators(self):
        if self.seed:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            np.random.seed(self.seed)
        self.train_iter, self.valid_iter, self.test_iter = Iterator.splits(datasets=(self.train_set, self.valid_set, self.test_set),
                                                                           batch_size=self.batch_size, shuffle=True, sort=False, device=self.device)