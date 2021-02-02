import math
import torch
from torch import nn
from model_crf import CRF
from transformers import AutoModel


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        ''' forward operation for network '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class NERModel(nn.Module):
    def __init__(self, input_dim, embedding_dim,
                 char_input_dim, char_embedding_dim,
                 char_filter, char_kernel, 
                 hidden_dim, output_dim, attn_heads, use_crf,
                 embedding_dropout_ratio, cnn_dropout_ratio, fc_dropout_ratio,
                 tag_names, text_pad_idx, text_unk_idx,
                 char_pad_idx, tag_pad_idx, pad_token,
                 pretrained_embeddings):
        '''

        basic class for named entity recognition models. inherits from neural network module.
        layers and forward function will be defined by a child class.

        input_dim: input dimension (size of text vocabulary)
        embedding_dim: embedding dimension (size of word vectors)
        char_input_dim: input dimension for characters (size of character vocabulary)
        char_embedding_dim: chatracter embedding dimension
        char_filter: number of filters for the character convolutions
        char_kernel: kernel size for character convolutions
        hidden_dim: hidden dimension
        output_dim: output dimension
        attn_heads: number of attention heads for attention component (set to zero or None to ignore)
        use_crf: switch for using conditional random field (reduces probability of invalid tagging sequences)
        embedding_dropout_ratio: dropout for embedding layer
        cnn_dropout_ratio: dropout for convolutions over characters
        fc_dropout_ratio: dropout for fully connected layer
        use_crf: switch for using conditional random field (reduces probability of invalid tagging sequences)
        tag_names: the names of all of the tags in the tag field
        text_pad_idx: index for text padding token
        text_unk_idx: indices for text unknown tokens
        char_pad_idx: indices for character unknown tokens
        tag_pad_idx: index for tag padding token
        pad_token: pad_token
        pretrained_embeddings: the pretrained word vectors for the dataset

        '''
        # initialize the superclass
        super().__init__()
        # dimensions
        self.bert_model = False
        self.input_dim, self.embedding_dim = input_dim, embedding_dim
        self.char_input_dim, self.char_embedding_dim = char_input_dim, char_embedding_dim
        self.char_filter, self.char_kernel = char_filter, char_kernel
        self.hidden_dim, self.output_dim = hidden_dim, output_dim
        # attention heads and crf
        self.attn_heads, self.use_crf = attn_heads, use_crf
        # dropout ratios
        self.embedding_dropout_ratio, self.cnn_dropout_ratio, self.fc_dropout_ratio = embedding_dropout_ratio, cnn_dropout_ratio, fc_dropout_ratio
        # tagging format
        self.tag_names = tag_names
        # indices for padding and unknown tokens
        self.text_pad_idx, self.text_unk_idx, self.char_pad_idx, self.tag_pad_idx = text_pad_idx, text_unk_idx, char_pad_idx, tag_pad_idx
        self.pad_token = pad_token
        # pretrained word embeddings
        self.pretrained_embeddings = pretrained_embeddings
    

    def init_weights(self):
        ''' initializes model weights '''
        for name, param in self.named_parameters():
            nn.init.normal_(param.data, mean=0, std=0.1)
    

    def init_embeddings(self):
        ''' initializes model embeddings  '''
        for idx in (self.text_unk_idx, self.text_pad_idx):
            self.embedding.weight.data[idx] = torch.zeros(self.embedding_dim)
        if self.pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(embeddings=torch.as_tensor(self.pretrained_embeddings), padding_idx=self.text_pad_idx, freeze=True)            


    def count_parameters(self):
        ''' counts model parameters '''
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BiLSTM_NER(NERModel):
    def __init__(self, input_dim, embedding_dim,
                 char_input_dim, char_embedding_dim,
                 char_filter, char_kernel, 
                 hidden_dim, output_dim,
                 lstm_layers, attn_heads, use_crf,
                 embedding_dropout_ratio, cnn_dropout_ratio, lstm_dropout_ratio,
                 attn_dropout_ratio, fc_dropout_ratio,
                 tag_names, text_pad_idx, text_unk_idx,
                 char_pad_idx, tag_pad_idx, pad_token,
                 pretrained_embeddings):
        '''

        BiLSTM model for named entity recognition. inherits from named recognition model

        input_dim: input dimension (size of text vocabulary)
        embedding_dim: embedding dimension (size of word vectors)
        char_input_dim: input dimension for characters (size of character vocabulary)
        char_embedding_dim: chatracter embedding dimension
        char_filter: number of filters for the character convolutions
        char_kernel: kernel size for character convolutions
        hidden_dim: hidden dimension
        output_dim: output dimension
        lstm_layers: number of lstm layers
        attn_heads: number of attention heads for attention component (set to zero or None to ignore)
        use_crf: switch for using conditional random field (reduces probability of invalid tagging sequences)
        embedding_dropout_ratio: dropout for embedding layer
        cnn_dropout_ratio: dropout for convolutions over characters
        lstm_dropout_ratio: dropout for lstm layers
        attn_dropout_ratio: dropout for attention layer
        fc_dropout_ratio: dropout for fully connected layer
        tag_names: the names of all of the tags in the tag field
        text_pad_idx: index for text padding token
        text_unk_idx: indices for text unknown tokens
        char_pad_idx: indices for character unknown tokens
        tag_pad_idx: index for tag padding token
        pad_token: pad_token
        pretrained_embeddings: the pretrained word vectors for the dataset

        '''
        # initialize the superclass
        super().__init__(input_dim, embedding_dim,
                         char_input_dim, char_embedding_dim,
                         char_filter, char_kernel, 
                         hidden_dim, output_dim, attn_heads, use_crf,
                         embedding_dropout_ratio, cnn_dropout_ratio, fc_dropout_ratio,
                         tag_names, text_pad_idx, text_unk_idx,
                         char_pad_idx, tag_pad_idx, pad_token,
                         pretrained_embeddings)
        # network structure settings
        self.lstm_layers = lstm_layers
        # dropout ratios
        self.lstm_dropout_ratio, self.attn_dropout_ratio = lstm_dropout_ratio, attn_dropout_ratio
        # build model layers
        self.build_model_layers()
        # initialize model weights
        self.init_weights()
        # initialize model embeddings
        self.init_embeddings()


    def build_model_layers(self):
        ''' builds the layers in the model '''
        # embedding layer
        self.embedding = nn.Embedding(num_embeddings=self.input_dim, embedding_dim=self.embedding_dim,
                                      padding_idx=self.text_pad_idx)
        # dropout for embedding layer
        self.embedding_dropout = nn.Dropout(self.embedding_dropout_ratio)
        # character cnn
        if self.char_embedding_dim:
            self.char_embedding = nn.Embedding(num_embeddings=self.char_input_dim,
                                               embedding_dim=self.char_embedding_dim,
                                               padding_idx=self.char_pad_idx)
            self.char_cnn = nn.Conv1d(in_channels=self.char_embedding_dim,
                                      out_channels=self.char_embedding_dim*self.char_filter,
                                      kernel_size=self.char_kernel,
                                      groups=self.char_embedding_dim)
            self.cnn_dropout = nn.Dropout(self.cnn_dropout_ratio)
            all_embedding_dim = self.embedding_dim+(self.char_embedding_dim*self.char_filter)
            # lstm layers with dropout
        else:
            all_embedding_dim = self.embedding_dim
        # lstm layers with dropout
        self.lstm = nn.LSTM(batch_first=True, input_size=all_embedding_dim,
                            hidden_size=self.hidden_dim, num_layers=self.lstm_layers,
                            bidirectional=True, dropout=self.lstm_dropout_ratio if self.lstm_layers > 1 else 0)
        # use multihead attention if there are attention heads
        if self.attn_heads:
            self.attn = nn.MultiheadAttention(embed_dim=self.hidden_dim*2, num_heads=self.attn_heads, dropout=self.attn_dropout_ratio)
        # dropout for fully connected layer
        self.fc_dropout = nn.Dropout(self.fc_dropout_ratio)
        # fully connected layer
        self.fc = nn.Linear(self.hidden_dim*2, self.output_dim)
        # use crf layer if it is switched on
        if self.use_crf:
            self.crf = CRF(self.tag_pad_idx, self.pad_token, self.tag_names)            
    

    def forward(self, sentence, characters, tags):
        ''' forward operation for network '''
        # the output of the embedding layer is dropout(embedding(input))
        embedding_out = self.embedding_dropout(self.embedding(sentence))
        if self.char_embedding_dim:
            char_embedding_out = self.embedding_dropout(self.char_embedding(characters))
            batch_size, sentence_len, word_len, char_embedding_dim = char_embedding_out.shape
            char_cnn_max_out = torch.zeros(batch_size, sentence_len, self.char_cnn.out_channels)
            # iterate over sentences
            for sentence_i in range(sentence_len):
                # character field of sentence i
                sentence_char_embedding = char_embedding_out[:, sentence_i, :, :]
                # channels last
                sentence_char_embedding_p = sentence_char_embedding.permute(0, 2, 1)
                char_cnn_sentence_out = self.char_cnn(sentence_char_embedding_p)
                char_cnn_max_out[:, sentence_i, :], _ = torch.max(char_cnn_sentence_out, dim=2)
            char_cnn = self.cnn_dropout(char_cnn_max_out)
            # concatenate word and character embeddings
            word_features = torch.cat((embedding_out, char_cnn), dim=2)
            # lstm of embedding output
            lstm_out, _ = self.lstm(word_features)
        else:
            # lstm of embedding output
            lstm_out, _ = self.lstm(embedding_out)
        # attention layer
        if self.attn_heads:
            # masking using the text padding index
            key_padding_mask = torch.as_tensor(sentence == self.text_pad_idx).permute(1, 0)
            # attention outputs
            attn_out, attn_weight = self.attn(lstm_out, lstm_out, lstm_out, key_padding_mask=key_padding_mask)
            # fully connected layer as function of attention output
            fc_out = self.fc(self.fc_dropout(attn_out))
        else:
            # fully connected layer as function of lstm output
            fc_out = self.fc(self.fc_dropout(lstm_out))
        if self.use_crf:
            crf_out, crf_loss = self.crf(fc_out, tags)
            return crf_out, crf_loss
        else:
            return fc_out


class Transformer_NER(NERModel):
    def __init__(self, input_dim, embedding_dim,
                 char_input_dim, char_embedding_dim,
                 char_filter, char_kernel, 
                 hidden_dim, output_dim,
                 trf_layers, attn_heads, use_crf,
                 embedding_dropout_ratio, cnn_dropout_ratio, trf_dropout_ratio,
                 fc_dropout_ratio,
                 tag_names, text_pad_idx, text_unk_idx,
                 char_pad_idx, tag_pad_idx, pad_token,
                 pretrained_embeddings):
        '''

        Transformer model for named entity recognition. inherits from neural network module

        input_dim: input dimension (size of text vocabulary)
        embedding_dim: embedding dimension (size of word vectors)
        char_input_dim: input dimension for characters (size of character vocabulary)
        char_embedding_dim: chatracter embedding dimension
        char_filter: number of filters for the character convolutions
        char_kernel: kernel size for character convolutions
        hidden_dim: hidden dimension
        output_dim: output dimension
        trf_layers: number of trf layers
        attn_heads: number of attention heads for attention component (set to zero or None to ignore)
        use_crf: switch for using conditional random field (reduces probability of invalid tagging sequences)
        embedding_dropout_ratio: dropout for embedding layer
        cnn_dropout_ratio: dropout for convolutions over characters
        trf_dropout_ratio: dropout for trf layers
        fc_dropout_ratio: dropout for fully connected layer
        tag_names: the names of all of the tags in the tag field
        text_pad_idx: index for text padding token
        text_unk_idx: indices for text unknown tokens
        char_pad_idx: indices for character unknown tokens
        tag_pad_idx: index for tag padding token
        pretrained_embeddings: the pretrained word vectors for the dataset

        '''
        # initialize the superclass
        super().__init__(input_dim, embedding_dim,
                         char_input_dim, char_embedding_dim,
                         char_filter, char_kernel, 
                         hidden_dim, output_dim, attn_heads, use_crf,
                         embedding_dropout_ratio, cnn_dropout_ratio, fc_dropout_ratio,
                         tag_names, text_pad_idx, text_unk_idx,
                         char_pad_idx, tag_pad_idx, pad_token,
                         pretrained_embeddings)
        # network structure settings
        self.trf_layers = trf_layers
        # dropout ratios
        self.trf_dropout_ratio = trf_dropout_ratio
        # build model layers
        self.build_model_layers()
        # initialize model weights
        self.init_weights()
        # initialize model embeddings
        self.init_embeddings()


    def build_model_layers(self):
        ''' builds the layers in the model '''
        # embedding layer
        self.embedding = nn.Embedding(num_embeddings=self.input_dim, embedding_dim=self.embedding_dim,
                                      padding_idx=self.text_pad_idx)
        # dropout for embedding layer
        self.embedding_dropout = nn.Dropout(self.embedding_dropout_ratio)
        # character cnn
        if self.char_embedding_dim:
            self.char_embedding = nn.Embedding(num_embeddings=self.char_input_dim,
                                               embedding_dim=self.char_embedding_dim,
                                               padding_idx=self.char_pad_idx)
            self.char_cnn = nn.Conv1d(in_channels=self.char_embedding_dim,
                                      out_channels=self.char_embedding_dim*self.char_filter,
                                      kernel_size=self.char_kernel,
                                      groups=self.char_embedding_dim)
            self.cnn_dropout = nn.Dropout(self.cnn_dropout_ratio)
            # lstm layers with dropout
            all_embedding_dim = self.embedding_dim+(self.char_embedding_dim*self.char_filter)
        else:
            all_embedding_dim = self.embedding_dim
        # transformer encoder layers with attention and dropout
        self.position_encoder = PositionalEncoding(d_model=all_embedding_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=all_embedding_dim, nhead=self.attn_heads, activation='relu', dropout=self.trf_dropout_ratio)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layers, num_layers=self.trf_layers)
        # fully connected layer with gelu activation
        self.fc1 = nn.Linear(in_features=all_embedding_dim, out_features=self.hidden_dim)
        self.fc1_gelu = nn.GELU()
        # layer norm
        self.fc1_norm = nn.LayerNorm(self.hidden_dim)
        # dropout for fully connected layer
        self.fc2_dropout = nn.Dropout(self.fc_dropout_ratio)
        # fully connected layer
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)
        # use crf layer if it is switched on
        if self.use_crf:
            self.crf = CRF(self.tag_pad_idx, self.pad_token, self.tag_names)
    

    def forward(self, sentence, characters, tags):
        ''' forward operation for network '''
        # the output of the embedding layer is dropout(embedding(input))
        embedding_out = self.embedding_dropout(self.embedding(sentence))
        key_padding_mask = torch.as_tensor(sentence == self.text_pad_idx).permute(1, 0)
        if self.char_embedding_dim:
            char_embedding_out = self.embedding_dropout(self.char_embedding(characters))
            batch_size, sentence_len, word_len, char_embedding_dim = char_embedding_out.shape
            char_cnn_max_out = torch.zeros(batch_size, sentence_len, self.char_cnn.out_channels)
            # iterate over sentences
            for sentence_i in range(sentence_len):
                # character field of sentence i
                sentence_char_embedding = char_embedding_out[:, sentence_i, :, :]
                # channels last
                sentence_char_embedding_p = sentence_char_embedding.permute(0, 2, 1)
                char_cnn_sentence_out = self.char_cnn(sentence_char_embedding_p)
                char_cnn_max_out[:, sentence_i, :], _ = torch.max(char_cnn_sentence_out, dim=2)
            char_cnn = self.cnn_dropout(char_cnn_max_out)
            # concatenate word and character embeddings
            word_features = torch.cat((embedding_out, char_cnn), dim=2)
            # positional encoding
            pos_out = self.position_encoder(word_features)
        else:
            # positional encoding
            pos_out = self.position_encoder(embedding_out)
        # encoding
        enc_out = self.encoder(pos_out, src_key_padding_mask=key_padding_mask)
        # fully connected layers
        fc1_out = self.fc1_norm(self.fc1_gelu(self.fc1(enc_out)))
        fc2_out = self.fc2(self.fc2_dropout(fc1_out))
        if self.use_crf:
            crf_out, crf_loss = self.crf(fc2_out, tags)
            return crf_out, crf_loss
        else:
            return fc2_out


class BERT_NER(nn.Module):
    def __init__(self, model_path, output_dim, use_crf, hidden_dropout_ratio, tag_pad_idx, pad_token, tag_names):
        '''
        bert sequence classifier
        output_dim: number of output classes
        use_crf: switch for using conditional random field (reduces probability of invalid tagging sequences)
        tag_pad_idx: index for tag padding token
        pad_token: pad token
        tag_names: the names of all of the tags in the tag field
        '''
        super().__init__()
        self.bert_model = True
        self.config = AutoConfig.from_pretrained(model_path)
        self.output_dim = output_dim
        self.use_crf = use_crf
        self.hidden_dropout_ratio = hidden_dropout_ratio
        self.tag_pad_idx, self.pad_token, self.tag_names = tag_pad_idx, pad_token, tag_names
        self.build_model_layers()
        self.init_weights()


    def build_model_layers(self):
        ''' builds the layers in the model '''
        self.bert = AutoModel.from_pretrained(model_path)
        self.fc_dropout = nn.Dropout(self.hidden_dropout_ratio)
        self.fc = nn.Linear(self.config.hidden_size, self.output_dim)
        if self.use_crf:
            self.crf = CRF(self.tag_pad_idx, self.pad_token, self.tag_names)
    

    def forward(self, sentence, attention_mask, tags):
        ''' forward operation for network '''
        outputs = self.bert(sentence, token_type_ids=None, attention_mask=attention_mask, labels=tags)
        sequence_output = outputs[0]
        logits = self.fc(self.fc_dropout(self.sequence_output))
        if self.use_crf:
            # remove first token id in each sentence (to make crf mask work)
            # crf_out, crf_loss = self.crf(logits, tags)
            crf_out, crf_loss = self.crf(logits[:, 1:], tags[:, 1:])
            return crf_out, crf_loss
        else:
            return logits
    

    def init_weights(self):
        ''' initializes model weights '''
        # param_initializer = list(self.bert.classifier.named_parameters())
        # if self.crf:
        #     param_initializer += list(self.crf.named_parameters())
        # for name, param in param_initializer:
        #     nn.init.normal_(param.data, mean=0, std=0.1)
        
        # only initialize conditional random field weights
        if self.crf:
            for name, param in self.crf.named_parameters():
                nn.init.normal_(param.data, mean=0, std=0.1)
        

    def count_parameters(self):
        ''' counts model parameters '''
        return sum(p.numel() for p in self.parameters() if p.requires_grad)