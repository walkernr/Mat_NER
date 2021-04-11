import torch
from torch import nn
import torchcrf

class CRF(nn.Module):
    def __init__(self, tag_pad_idx, pad_token, tag_names, tag_format):
        super().__init__()
        # tag pad index and tag names
        self.tag_pad_idx = tag_pad_idx
        self.pad_token = pad_token
        self.tag_names = tag_names
        self.tag_format = tag_format
        self.prefixes = set([tag_name[0] for tag_name in self.tag_names if tag_name != self.pad_token])
        # initialize CRF
        self.crf = torchcrf.CRF(num_tags=len(self.tag_names), batch_first=True)
    

    def define_invalid_crf_transitions(self):
        ''' function for establishing valid tagging transitions, assumes IOB1, IOB2, or IOBES tagging '''
        if self.tag_format == 'IOB1':
            # (B)eginning (I)nside (O)utside
            # all beginnings are valid
            self.invalid_begin = ()
            # cannot end sentence with B (beginning)
            self.invalid_end = ('B',)
            # prevent B (beginning) going to B (beginning) or O (outside) - B must be followed by I
            self.invalid_transitions_position = {'B': 'BO'}
            # prevent B (beginning) going to I (inside) or B (beginning) of a different type
            self.invalid_transitions_tags = {'B': 'IB'}
        elif self.tag_format == 'IOB2':
            # (B)eginning (I)nside (O)utside
            # cannot begin sentence with I (inside), only B (beginning) or O (outside)
            self.invalid_begin = ('I',)
            # all endings are valid
            self.invalid_end = ()
            # prevent O (outside) going to I (inside) - O must be followed by B or O
            self.invalid_transitions_position = {'O': 'I'}
            # prevent B (beginning) going to I (inside) of a different type
            # prevent I (inside) going to I (inside) of a different type
            self.invalid_transitions_tags = {'B': 'I',
                                             'I': 'I'}
        elif self.tag_format == 'IOBES':
            # (I)nside (O)utside (B)eginning (E)nd (S)ingle 
            # cannot begin sentence with I (inside) or E (end)
            if 'I' in self.prefixes:
                self.invalid_begin = ('I', 'E')
            else:
                self.invalid_begin = ('E',)
            # cannot end sentence with B (beginning) or I (inside)
            if 'I' in self.prefixes:
                self.invalid_end = ('B', 'I')
            else:
                self.invalid_end = ('B',)
            # prevent B (beginning) going to B (beginning), O (outside), or S (single) - B must be followed by I or E
            # prevent I (inside) going to B (beginning), O (outside), or S (single) - I must be followed by I or E
            # prevent E (end) going to I (inside) or E (end) - U must be followed by B, O, or U
            # prevent S (single) going to I (inside) or E (end) - U must be followed by B, O, or U
            # prevent O (outside) going to I (inside) or E (end) - O must be followed by B, O, or U
            if 'I' in self.prefixes:
                self.invalid_transitions_position = {'B': 'BOS',
                                                     'I': 'BOS',
                                                     'E': 'IE',
                                                     'S': 'IE',
                                                     'O': 'IE'}
            else:
                self.invalid_transitions_position = {'B': 'BOS',
                                                     'E': 'E',
                                                     'S': 'E',
                                                     'O': 'E'}
            # prevent B (beginning) from going to I (inside) or E (end) of a different type
            # prevent I (inside) from going to I (inside) or E (end) of a different tpye
            if 'I' in self.prefixes:
                self.invalid_transitions_tags = {'B': 'IE',
                                                 'I': 'IE'}
            else:
                self.invalid_transitions_tags = {'B': 'E'}
    

    def init_crf_transitions(self, imp_value=-10000):
        num_tags = len(self.tag_names)
        # penalize bad beginnings and endings
        for i in range(num_tags):
            tag_name = self.tag_names[i]
            if tag_name[0] in self.invalid_begin or tag_name == self.pad_token:
                torch.nn.init.constant_(self.crf.start_transitions[i], imp_value)
            if tag_name[0] in self.invalid_end:
                torch.nn.init.constant_(self.crf.end_transitions[i], imp_value)
        # build tag type dictionary
        tag_is = {}
        for tag_position in self.prefixes:
            tag_is[tag_position] = [i for i, tag in enumerate(self.tag_names) if tag[0] == tag_position]
        # penalties for invalid consecutive tags by position
        for from_tag, to_tag_list in self.invalid_transitions_position.items():
            to_tags = list(to_tag_list)
            for from_tag_i in tag_is[from_tag]:
                for to_tag in to_tags:
                    for to_tag_i in tag_is[to_tag]:
                        torch.nn.init.constant_(self.crf.transitions[from_tag_i, to_tag_i], imp_value)
        # penalties for invalid consecutive tags by tag
        for from_tag, to_tag_list in self.invalid_transitions_tags.items():
            to_tags = list(to_tag_list)
            for from_tag_i in tag_is[from_tag]:
                for to_tag in to_tags:
                    for to_tag_i in tag_is[to_tag]:
                        if self.tag_names[from_tag_i].split('-')[1] != self.tag_names[to_tag_i].split('-')[1]:
                            torch.nn.init.constant_(self.crf.transitions[from_tag_i, to_tag_i], imp_value)        
    

    def forward(self, fc_out, tags):
        # mask ignores pad index
        mask = (tags != self.tag_pad_idx)
        # compute output and loss
        crf_out = self.crf.decode(fc_out, mask=mask)
        crf_loss = -self.crf(fc_out, tags=tags, mask=mask, reduction='mean')
        return crf_out, crf_loss
