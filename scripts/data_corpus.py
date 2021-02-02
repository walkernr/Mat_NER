from collections import Counter
import gensim
import torch
from torchtext.data import Field, NestedField, BucketIterator
from torchtext.datasets import SequenceTaggingDataset
from torchtext.vocab import Vocab


class DataCorpus(object):
    def __init__(self, data_path, data_name, alias, vector_path, tokenizer, max_sequence_length, cased, bert, batch_size, device):
        self.data_path = data_path
        self.data_name = data_name
        self.alias = alias
        self.vector_path = vector_path
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.cased = cased
        self.bert = bert
        self.batch_size = batch_size
        self.device = device
        self.pad_token = self.tokenizer.pad_token
        self.unk_token = self.tokenizer.unk_token
        self.initialize_fields()
        self.load_data()
        self.build_tag_vocabulary()
        self.build_text_vocabulary()
        self.build_char_vocabulary()
        self.initialize_iterators()


    def insert_initial_pad_tag(self, tag):
        tag.insert(0, self.pad_token)
        return tag


    def initialize_fields(self):
        if self.bert:
            self.text_field = Field(use_vocab=False, preprocessing=self.tokenizer.encode, lower=not self.cased,
                                    pad_token=self.tokenizer.convert_tokens_to_ids(self.pad_token),
                                    unk_token=self.tokenizer.convert_tokens_to_ids(self.unk_token),
                                    fix_length=self.max_sequence_length, include_lengths=False, batch_first=True)
            self.tag_field = Field(preprocessing=self.insert_initial_pad_tag,
                                   pad_token=self.pad_token, unk_token=None,
                                   fix_length=self.max_sequence_length, batch_first=True)
        else:
            self.text_field = Field(tokenize=self.tokenizer.tokenize, preprocessing=self.tokenizer.process, lower=not self.cased,
                                    pad_token=self.pad_token, unk_token=self.unk_token,
                                    batch_first=True)
            self.tag_field = Field(pad_token=self.pad_token, unk_token=None,
                                   batch_first=True)
            char_nesting_field = Field(tokenize=list, pad_token=self.pad_token, batch_first=True)
            self.char_field = NestedField(char_nesting_field)

    
    def load_data(self):
        fields = ((('text', 'char'), (self.text_field, self.char_field)),
                  ('tag', self.tag_field))
        self.train_set, self.valid_set, self.test_set = SequenceTaggingDataset.splits(fields=fields, path=self.data_path+'/split/',
                                                                                      train=self.data_name+self.alias+'_train.tsv',
                                                                                      validation=self.data_name+self.alias+'_valid.tsv',
                                                                                      test=self.data_name+self.alias+'_test.tsv')


    def build_tag_vocabulary(self):
        self.tag_field.build_vocab(self.train_set.tag)
        self.tag_pad_idx = self.tag_field.vocab.stoi[self.pad_token]
        self.tag_names = self.tag_field.vocab.itos
        prefixes = set([tag_name[0] for tag_name in self.tag_names if tag_name != self.pad_token])
        if prefixes == set(['B', 'I', 'O']):
            self.format = 'IOB2'
        elif prefixes == set(['B', 'I', 'L', 'U', 'O']):
            self.format = 'BILOU'
        elif prefixes == set(['B', 'I', 'E', 'S', 'O']):
            self.format = 'BIOES'


    def build_text_vocabulary(self):
        if not self.bert:
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
        else:
            pass

    
    def build_char_vocabulary(self):
        if not self.bert:
            self.char_field.build_vocab(self.train_set.char)
            self.char_pad_idx = self.char_field.vocab.stoi[self.pad_token]
        else:
            pass
    

    def initialize_iterators(self):
        self.train_iter, self.valid_iter, self.test_iter = BucketIterator.splits(datasets=(self.train_set, self.valid_set, self.test_set),
                                                                                 batch_size=self.batch_size, sort=False, device=self.device)