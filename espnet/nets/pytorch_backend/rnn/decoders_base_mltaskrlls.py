#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# this code is for end-2end speech recognition;(based on espnet)
#
# Authors: Jian Tang (modification)
'''
Build a encoder&decdoer model with soft attention
    Nums = 315
'''
from distutils.version import LooseVersion
import logging
import random
import six

import numpy as np
import torch
import torch.nn.functional as F

from argparse import Namespace

from espnet.nets.ctc_prefix_score_old import CTCPrefixScore # org : from espnet.nets.ctc_prefix_score import CTCPrefixScore
from espnet.nets.ctc_prefix_score_old import CTCPrefixScoreTH # org : from espnet.nets.ctc_prefix_score import CTCPrefixScoreTH
from espnet.nets.e2e_asr_common import end_detect

from espnet.nets.pytorch_backend.rnn.attentions import att_to_numpy
from espnet.nets.pytorch_backend.nets_utils import append_ids
from espnet.nets.pytorch_backend.nets_utils import get_last_yseq
from espnet.nets.pytorch_backend.nets_utils import index_select_list
from espnet.nets.pytorch_backend.nets_utils import index_select_lm_state
from espnet.nets.pytorch_backend.nets_utils import mask_by_length
from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import make_trueframe_mask

MAX_DECODER_OUTPUT = 5 # ?
CTC_SCORING_RATIO = 1.5 # ?
traditional_loss_calculate = False
decode_choice = "bpe"
# CHOICE_SCORE_TYPES = "bpe2char" # "bpe1char" / "bpe2char" "bpeonly"
switch_addcode_forbi = True # False is same as mltaskrll / True is BIDIR (mltaskrll + d)

def slow_onehot(labels, senone=52):
    if len(labels.size())==2: # not finished yet
        batch, timestep = labels.size()
        one_hot = torch.zeros(batch, timestep, senone)
        print('This code have not applied yet.')
        exit(0)
    elif len(labels.size())==1:
        y = torch.eye(senone) 
        mask = 1 - torch.eq(labels,-1).to(labels).float()
        return ((y[labels].to(labels)).float())*mask.unsqueeze(1)


def kullback_leibler_divergence(y_true, y_pred, axis=-1): 
    y_true = torch.clamp(y_true, 1e-8, 1)
    y_pred = torch.clamp(y_pred, 1e-8, 1)
    return torch.sum(y_true * torch.log(y_true / y_pred), dim=axis)


class MLSoftmaxDecoder(torch.nn.Module):
    def __init__(self, eprojs, dtype, dlayers, dunits, 
                    odim_char, odim_bpe, 
                    sos_char, eos_char, sos_bpe, eos_bpe, 
                    att_char, att_bpe, att_char2bpe, 
                    args, 
                    verbose=0, char_list=None, bpe_list=None, 
                    labeldist_char=None, labeldist_bpe=None, 
                    lsm_weight=0., sampling_probability=0.0, dropout=0.0,
                    embsize_char=128, embsize_bpe=1000): ## RIGHT(28-28)
        super(MLSoftmaxDecoder, self).__init__()
        self.dtype = dtype
        self.dunits = dunits
        self.dlayers = dlayers

        ###### Parameters for CHAR decoder and the additional operation (start)  ######
        ## embedding layer
        self.embsize_char = embsize_char
        self.embed_char = torch.nn.Embedding(odim_char, self.embsize_char)
        self.dropout_emb_char = torch.nn.Dropout(p=dropout)
        ## Split RNN Information Combination Layer ##
        self.char_rnn1_Wx    = torch.nn.Linear(self.embsize_char, dunits)
        self.char_rnn1_W     = torch.nn.Linear(self.embsize_char, dunits*2)
        self.char_rnn1_U     = torch.nn.Linear(dunits, dunits*2, bias=False)
        self.char_rnn1_Ux    = torch.nn.Linear(dunits, dunits, bias=False)
        ## Split RNN for next iterator layer ##
        self.char_rnn2_U_n1  = torch.nn.Linear(dunits, dunits*2)
        self.char_rnn2_Wc    = torch.nn.Linear(eprojs, dunits*2, bias=False)
        self.char_rnn2_Ux_n1 = torch.nn.Linear(dunits, dunits)
        self.char_rnn2_Wcx   = torch.nn.Linear(eprojs, dunits, bias=False)
        ## New added here by jtang ##
        self.init_char_state_fc = torch.nn.Linear(eprojs, dunits)
        ## Classification Part ##
        self.classchar_state_W = torch.nn.Linear(dunits, dunits)
        self.classchar_yprev_W = torch.nn.Linear(self.embsize_char, dunits, bias=False)
        self.classchar_enc_W   = torch.nn.Linear(eprojs, dunits, bias=False)
        self.char_class_W    = torch.nn.Linear(dunits//2, odim_char)
        ## Constant value
        self.sos_char, self.eos_char = sos_char, eos_char
        self.odim_char = odim_char
        self.att_char = att_char
        ###### Parameters for CHAR decoder and the additional operation (finish) ######


        ###### New Parameters added for Bidirection (char->BPE && BPE->char start) ######
        if switch_addcode_forbi == True:
            # self.bpe2char_bn1D = torch.nn.BatchNorm1d(dunits)
            # self.bpe2char_bn2D = torch.nn.BatchNorm1d(dunits)
            self.classb2c_state_W1 = torch.nn.Linear(dunits, dunits)             
            self.classb2c_state_W2 = torch.nn.Linear(dunits, dunits)
        ###### New Parameters added for Bidirection (char->BPE && BPE->char finish) ######

        print('The decoder version is multi-level char&bpe (code checking -> second joint decoding).')
        print('Setting loss calc', traditional_loss_calculate)
        print('The lsm_weight value is %f' %(lsm_weight))
        print('Deep Fusion :  v10rd... [h1;sigmoid(FC(h1))*h3].')
        print('The embsize_bpe is %d while embsize_char is %d.' %(embsize_bpe, embsize_char))
        print('******************No lambdacharloss (1.0 stable) here org.******************')
        if switch_addcode_forbi == True: # False is same as mltaskrll / True is BIDIR (mltaskrll + d)
            print('[decoders_base_mltaskrlld.py] This is just information transformation from char to BPE (mltaskbaserlls).')
            print('The decoder version is multi-level char&bpe (BIDIR : code checking -> second joint decoding).')
            print('残差作用于BPE -- CHAR 的效果： W1-BN-ReLU-W2-BN 构成支线')
        else:
            print('[decoders_base_mltaskrll.py] This is just information transformation from char to BPE (mltaskbaserll).')
            print('The decoder version is multi-level char&bpe (UNIDIR : code checking -> second joint decoding).')        

        ###### Parameters for BPE decoder and the additional operation (start)  ######
        cprojs = eprojs #// 2
        ## embedding layer
        self.embsize_bpe = embsize_bpe
        self.embed_bpe = torch.nn.Embedding(odim_bpe, self.embsize_bpe)
        self.dropout_emb_bpe = torch.nn.Dropout(p=dropout)
        
        ## BPE RNN Layer 1
        self.bpe_rnn1_Wx = torch.nn.Linear(self.embsize_bpe, dunits)
        self.bpe_rnn1_W  = torch.nn.Linear(self.embsize_bpe, dunits*2)
        self.bpe_rnn1_U  = torch.nn.Linear(dunits, dunits*2, bias=False)
        self.bpe_rnn1_Ux = torch.nn.Linear(dunits, dunits, bias=False)
        ## BPE RNN Layer 2
        self.bpe_rnn2_U_n1  = torch.nn.Linear(dunits, dunits*2)
        self.bpe_rnn2_Wc    = torch.nn.Linear(cprojs, dunits*2, bias=False)
        self.bpe_rnn2_Ux_n1 = torch.nn.Linear(dunits, dunits)
        self.bpe_rnn2_Wcx   = torch.nn.Linear(cprojs, dunits, bias=False)

        self.att_char2bpe = att_char2bpe
        ## CHAR2BPE RNN Layer 1
        self.char2bpe_rnn1_Wx = torch.nn.Linear(self.embsize_bpe, dunits)
        self.char2bpe_rnn1_W  = torch.nn.Linear(self.embsize_bpe, dunits*2)
        self.char2bpe_rnn1_U  = torch.nn.Linear(dunits, dunits*2, bias=False)
        self.char2bpe_rnn1_Ux = torch.nn.Linear(dunits, dunits, bias=False)
        ## CHAR2BPE RNN Layer 2
        self.c2b_statecomb = torch.nn.Linear(dunits, dunits)
        self.char2bpe_rnn2_U_n1  = torch.nn.Linear(dunits, dunits*2)
        self.char2bpe_rnn2_Wc    = torch.nn.Linear(cprojs, dunits*2, bias=False)
        self.char2bpe_rnn2_Ux_n1 = torch.nn.Linear(dunits, dunits)
        self.char2bpe_rnn2_Wcx   = torch.nn.Linear(cprojs, dunits, bias=False)

        ## New added here by jtang ##
        self.init_bpe_state_fc = torch.nn.Linear(cprojs, dunits)
        ## BPE classification later
        self.bpeclass_dunits = 1024
        self.classbpe_state_W = torch.nn.Linear(2*dunits, 2*self.bpeclass_dunits)
        self.classbpe_yprev_W = torch.nn.Linear(self.embsize_bpe, 2*self.bpeclass_dunits, bias=False)
        self.classbpe_enc_W   = torch.nn.Linear(2*cprojs, 2*self.bpeclass_dunits, bias=False)
        self.bpe_class_W = torch.nn.Linear(self.bpeclass_dunits, odim_bpe)
        ## Constant value
        self.sos_bpe, self.eos_bpe = sos_bpe, eos_bpe
        self.odim_bpe = odim_bpe
        self.att_bpe = att_bpe
        
        ###### Parameters for BPE decoder and the additional operation (finish) ######

        #########################
        self.ignore_id = -1
        self.loss = None
        #########################

        self.dunits = dunits
        self.verbose = verbose
        
        # for label smoothing 
        self.labeldist_char, self.labeldist_bpe = labeldist_char, labeldist_bpe
        self.vlabeldist_char, self.vlabeldist_bpe = None, None # ?
        self.lsm_weight = lsm_weight 
        self.sampling_probability = sampling_probability # focus 0 means all true-label 
        self.dropout = dropout

        #########################
        if bpe_list is None:
            bpe_list = "data/lang_char/train_nodup_bpe2000_units.txt"
            with open(bpe_list, 'rb') as f:
                dictionary = f.readlines()
            bpe_list = [entry.decode('utf-8').split(' ')[0]
                         for entry in dictionary]
            bpe_list.insert(0, '<blank>')
            bpe_list.append('<eos>')
        
        self.bpedict = {}
        for i in bpe_list: self.bpedict[bpe_list.index(i)] = i  
        self.bpedict[-1] = '<padding>' 
        #########################

        #########################
        if char_list is None:
            char_list = "data/lang_1char/train_nodup_units.txt"
            with open(char_list, 'rb') as f:
                dictionary = f.readlines()
            char_list = [entry.decode('utf-8').split(' ')[0]
                         for entry in dictionary]
            char_list.insert(0, '<blank>')
            char_list.append('<eos>')
        
        self.chardict = {}
        for i in char_list: self.chardict[char_list.index(i)] = i   
        
        self.inverse_chardict=dict([val,key] for key,val in self.chardict.items())  
        val = self.inverse_chardict['<space>']
        self.inverse_chardict.pop('<space>')
        self.inverse_chardict['▁'] = val
        #########################        

        self.char_list = char_list
        self.bpe_list = bpe_list

        self.char_eosmask = None
        self.logzero = -10000000000.0

        self.batchcount = 0

    # initial decoder state  enc_o[B,T,D] -> mean -> (B,D_enc) -> FC -> [B,D_dec]
    def mean_init_char_state(self, hs_pad, hlens):
        hlens = (torch.Tensor(hlens)).float().to(hs_pad)                             # training process
        mean_enc_o = torch.sum(input=hs_pad, dim=1)  # [B,D_enc]
        result = torch.div( mean_enc_o, hlens.view(hs_pad.size(0), 1) ) # [B,1]
        result = self.init_char_state_fc(result)
        return result


    def mean_init_bpe_state(self, hs_pad, hlens):
        hlens = (torch.Tensor(hlens)).float().to(hs_pad)                             # training process
        mean_enc_o = torch.sum(input=hs_pad, dim=1)  # [B,D_enc]
        result = torch.div( mean_enc_o, hlens.view(hs_pad.size(0), 1) ) # [B,1]
        result = self.init_bpe_state_fc(result)
        return result        


    def mean_init_state_decoder(self, hs): # seq_len, eprojs = hs.size()
        mean_enc_o = torch.mean(input=hs, dim=0)  # [D]
        result = self.init_state_fc(mean_enc_o)
        return result        


    def zero_state(self, hs_pad):
        return hs_pad.new_zeros(hs_pad.size(0), self.dunits)


    def _slice(self, _x, n, dim):
        if len(_x.size()) == 3: 
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]


    # Forward Process
    def forward(self, hs_pad, hlens, ys_pad_char, ys_pad_bpe, strm_idx=0): 
        """Decoder forward

            :param torch.Tensor hs_pad: batch of padded hidden state sequences (B, Tmax, D) : enc_o
            :param torch.Tensor hlens: batch of lengths of hidden state sequences (B) : enc_o_len
            :param torch.Tensor ys_pad1: batch of padded token id sequence tensor (B, Lmax1) : char true_label
            :param torch.Tensor ys_pad1: batch of padded token id sequence tensor (B, Lmax2) : bpe true_label
            :param int strm_idx: stream index indicates the index of decoding stream. (y_idx ? )
            :return: attention loss value
            :rtype: torch.Tensor
            :return: accuracy
            :rtype: float
        """
        SWITCH_BPE = True
        SWITCH_CHAR = True
        self.batchcount += 1        
        nbatch = hs_pad.shape[0]

        if decode_choice == "bpe":
            ###### 0.Codes for CHAR decoder and the additional operation (start)  ######
            # initialization : give initial context_vector & decoder_state
            
            # TODO(kan-bayashi): need to make more smart way
            ys_char = [y[y != self.ignore_id] for y in ys_pad_char]  # parse padded ys (char) List

            # attention index for the attention module in SPA (speaker parallel attention), att_idx is used to select attention module. In other cases, it is 0.
            att_char_idx = min(strm_idx, len(self.att_char) - 1) # useful only for SAT

            # reset pre-computation of h
            self.att_char[att_char_idx].reset()

            # prepare input and output word sequences with sos/eos IDs (char)
            eos_char = ys_char[0].new([self.eos_char])
            sos_char = ys_char[0].new([self.sos_char])
            ys_char_in = [torch.cat([sos_char, y], dim=0) for y in ys_char] # in this way : the label and the feedback y is created (ys_in is for state(t-1))
            ys_char_out = [torch.cat([y, eos_char], dim=0) for y in ys_char] #  

            # char y-label length
            ys_char_out_len = [y.size(0) for y in ys_char_out]
            y_char_mask = (to_device(self, make_trueframe_mask(ys_char_out_len))).float()

            # input mask
            context_mask = (to_device(self, make_trueframe_mask(hlens))).float() # [B,T]=[10,203] torch.Tensor

            # padding for ys with -1
            # pys: utt x olen (char)
            ys_char_in_pad  = pad_list(ys_char_in, self.eos_char).to(hs_pad).long() # pad feature torch.Tensor([B,decT])
            ys_char_out_pad = pad_list(ys_char_out, self.ignore_id).to(hs_pad).long() # pad output symbol

            # pre-computation of embedding y -> emb(y) [B,C,D_senone]
            eys_char = self.dropout_emb_char(self.embed_char(ys_char_in_pad))  # utt x olen x zdim [B,decoder_T,300]

            # hlen should be list of integer
            hlens_char = list(map(int, hlens))

            # get dim, length info [B,T_dec] = [15,168]
            olength_char = ys_char_out_pad.size(1)
            logging.info(self.__class__.__name__ + ' input(speech) lengths:  ' + str(hlens_char))
            logging.info(self.__class__.__name__ + ' output(char) lengths: ' + str([y.size(0) for y in ys_char_out]))

            # Unchanged varibles
            char_enc = hs_pad
            subh_ = self.mean_init_char_state(char_enc, hlens_char)                         #.repeat(n_bb, 1) # [n_bb, dunits]
            subprev_energy = (to_device(self, make_trueframe_mask(hlens_char))).float()     # previous postr attention score
            subprev_alpha = ( subprev_energy / (subprev_energy.sum(1, keepdim=True)+1e-8) ) # [n_bb, ]           

            char_prior_att_weights = []
            char_prior_att_context = []

            # This is for the char decoder process
            state_char  = eys_char                   # y(t-1): [B,decT,embsize_char]=[B,144,256]
            state_char_ = self.char_rnn1_W(eys_char)  # [B,decT,dunits]=[10,144,600]
            state_charx = self.char_rnn1_Wx(eys_char) # [B,decT,dunits]=[10,144,300]
            ###### 0.Codes for CHAR decoder and the additional operation (finish) ######       

        if SWITCH_BPE == True:
            ###### Codes for BPE decoder and the additional operation (start)  ######
            # initialization : give initial context_vector & decoder_state
            
            # TODO(kan-bayashi): need to make more smart way
            ys_bpe = [y[y != self.ignore_id] for y in ys_pad_bpe]  # parse padded ys (bpe) List

            # attention index for the attention module in SPA (speaker parallel attention), att_idx is used to select attention module. In other cases, it is 0.
            att_bpe_idx = min(strm_idx, len(self.att_bpe) - 1) # useful only for SAT

            # reset pre-computation of h
            self.att_bpe[att_bpe_idx].reset()
            self.att_char2bpe[att_bpe_idx].reset()

            # prepare input and output word sequences with sos/eos IDs (bpe)
            eos_bpe = ys_bpe[0].new([self.eos_bpe])
            sos_bpe = ys_bpe[0].new([self.sos_bpe])
            ys_bpe_in = [torch.cat([sos_bpe, y], dim=0) for y in ys_bpe] # in this way : the label and the feedback y is created (ys_in is for state(t-1))
            ys_bpe_out = [torch.cat([y, eos_bpe], dim=0) for y in ys_bpe] # 

            # print('Start Decoding Process. 考虑一下如果是ignore的情况怎么消除')
            ys_bpe_all = (pad_list(ys_bpe_out, self.ignore_id)).transpose(1, 0).contiguous().tolist() # Torch.tensor((batch,decT)) : -1 means useless
            decTlen = len(ys_bpe_all)

            ysbpe_subseqs = [[] for _ in range(decTlen)]
            char_len = np.zeros((decTlen, nbatch), dtype=np.int)

            yseq_char_inbpe = [[] for _ in range(decTlen)]
            yseq_char_inbpe[0] = [[self.sos_char] for y in six.moves.range(nbatch)] ## 只是为了循环体的变量

            # print('准备好每一步转换对应的char子序列情况：后面我还是会验证上面的准备数据结果的')
            for trans_idx, yseqs in enumerate(ys_bpe_all): # yseqs list(batch)  (decT(*),B)
                bpe_token_list = [self.bpedict[idx] for idx in yseqs] # ['▁ALSO']

                for idx,bpe_token in enumerate(bpe_token_list):
                    if ((trans_idx == 0) and bpe_token == "▁") or (bpe_token == "<padding>"): ## 如果第一个字符是"▁", 那么：就选择忘记这个BPE的情况continue:
                        char_len[trans_idx,idx] = len(yseq_char_inbpe[trans_idx][idx]) - 1
                        continue                    
                    elif (trans_idx == 0) and (bpe_token[0] == "▁"):
                        token = bpe_token[1:]
                    else:
                        token = bpe_token
                    
                    if token == '[laughter]' or token == '[noise]' or token == '[vocalized-noise]' or token == '<eos>' or token == '<blank>' or token == '<unk>': #   <unk> <blank> <eos>
                        yseq_char_inbpe[trans_idx][idx].append(self.inverse_chardict[token]) ## all length
                    else:
                        for iidx, str_ in enumerate(token):
                            yseq_char_inbpe[trans_idx][idx].append(self.inverse_chardict[str_]) ## all length

                    char_len[trans_idx,idx] = len(yseq_char_inbpe[trans_idx][idx]) - 1

                if trans_idx+1 < decTlen: yseq_char_inbpe[trans_idx+1] = [[chartoken[-1]] for chartoken in yseq_char_inbpe[trans_idx]]

            # bpe y-label length
            ys_bpe_out_len = [y.size(0) for y in ys_bpe_out]
            y_bpe_mask = (to_device(self, make_trueframe_mask(ys_bpe_out_len))).float()

            # input mask
            context_mask = (to_device(self, make_trueframe_mask(hlens))).float() # [B,T]=[10,203] torch.Tensor

            # padding for ys with -1
            # pys: utt x olen (bpe)
            ys_bpe_in_pad  = pad_list(ys_bpe_in, self.eos_bpe).to(hs_pad).long() # pad feature torch.Tensor([B,decT])
            ys_bpe_out_pad = pad_list(ys_bpe_out, self.ignore_id).to(hs_pad).long() # pad output symbol

            # pre-computation of embedding y -> emb(y) [B,C,D_senone]
            eys_bpe = self.dropout_emb_bpe(self.embed_bpe(ys_bpe_in_pad))  # utt x olen x zdim [B,decoder_T,300]

            # hlen should be list of integer
            hlens_bpe = list(map(int, hlens))

            # get dim, length info [B,T_dec] = [15,168]
            batch = ys_bpe_out_pad.size(0)
            olength_bpe = ys_bpe_out_pad.size(1)
            logging.info(self.__class__.__name__ + ' input(speech) lengths:  ' + str(hlens_bpe))
            logging.info(self.__class__.__name__ + ' output(bpe) lengths: ' + str([y.size(0) for y in ys_bpe_out]))

            # Unchanged varibles
            bpe_enc = hs_pad
            h_ = self.mean_init_bpe_state(bpe_enc, hlens_bpe) # hlens_bpe ?
            hbpe_ = self.mean_init_bpe_state(bpe_enc, hlens_bpe) # hlens_bpe ?
            prev_postr_energy = (to_device(self, make_trueframe_mask(hlens_bpe))).float() # previous postr attention score
            prev_alpha = prev_postr_energy / (prev_postr_energy.sum(1, keepdim=True)+1e-8) # [B,T]      
            prev_c2b_alpha = prev_alpha.clone()      
            
            bpe_prior_att_weights = []
            bpe_prior_att_context = []
            bpe_output_all = []       
            c2b_prior_att_weights = []
            c2b_prior_att_context = []             

            # This is for the bpe decoder process
            state_bpe  = eys_bpe                   # y(t-1): [B,decT,embsize_bpe]=[B,144,256]
            state_bpe_ = self.bpe_rnn1_W(eys_bpe)  # [B,decT,dunits]=[10,144,600]
            state_bpex = self.bpe_rnn1_Wx(eys_bpe) # [B,decT,dunits]=[10,144,300]
            charposterindex = np.zeros(nbatch, dtype=np.int)
            char_output_all = (1.0/self.odim_char) * torch.ones((nbatch, olength_char, self.odim_char)).to(hs_pad)

            state_char2bpe_ = self.char2bpe_rnn1_W(eys_bpe)  # [B,decT,dunits]=[10,144,600]
            state_char2bpex = self.char2bpe_rnn1_Wx(eys_bpe) # [B,decT,dunits]=[10,144,300]
            for i in six.moves.range(olength_bpe):
                m_  = y_bpe_mask[:,i] # the variables for this bpe decoder steps
                x_  = state_bpe_[:,i,:]
                xx_ = state_bpex[:,i,:]
                cc_ = bpe_enc # [B,decT,dunit//2]
                c2b_x_  = state_char2bpe_[:,i,:]
                c2b_xx_ = state_char2bpex[:,i,:]                

                ## 1. RNN : s'(t)=RNN(y(t-1),s(t-1)) 
                # || to combine [emb(t-1),state(t-1)] -> predt-state(t) 
                # || RNN(dunits) : concat[ey,ctx-previous-posterior]
                if switch_addcode_forbi == True: infor_bpe2char = h_
                preact1 = self.bpe_rnn1_U(h_) # no bias -> [B,dunits*2]=[10,600]
                preact1 += x_ # [B,dunits*2]=[10,600]
                preact1 = torch.sigmoid(preact1)
                r1 = self._slice(preact1, 0, self.dunits) # reset gate [B,dunits]=[10,300]
                u1 = self._slice(preact1, 1, self.dunits) # update gate [B,dunits]=[10,300]
                preactx1 = self.bpe_rnn1_Ux(h_) # no bias [B,dunits]=[10,300]
                preactx1 *= r1
                preactx1 += xx_ 
                h1 = torch.tanh(preactx1)
                h1 = u1 * h_ + (1. - u1) * h1
                h1 = m_.unsqueeze(1) * h1 + (1. - m_).unsqueeze(1) * h_

                ## 2. Attention : c(t)=AttentionContext(s'(t), enc_o) 
                # || dunits (prior_att_c[B,eprojs] prior_att_w[B,T])
                prior_att_c, prior_att_w = self.att_bpe[att_bpe_idx](cc_, hlens_bpe, h1, prev_alpha)
                bpe_prior_att_weights.append(prior_att_w)
                bpe_prior_att_context.append(prior_att_c)

                ## 3. RNN : s(t)=RNN(c(t),s'(t)) 
                # || att_c(eprojs) z_infocomb_list(dunits)
                preact2 = self.bpe_rnn2_U_n1(h1) # [B,dunits*2]=[10,600]
                preact2 += self.bpe_rnn2_Wc(prior_att_c) # no bias [B,dunits*2]=[10,600]
                preact2 = torch.sigmoid(preact2)
                r2 = self._slice(preact2, 0, self.dunits) # [B,dunits]
                u2 = self._slice(preact2, 1, self.dunits) # [B,dunits]
                preactx2 = self.bpe_rnn2_Ux_n1(h1) # [B,dunits]=[10,300]
                preactx2 *= r2
                preactx2 += self.bpe_rnn2_Wcx(prior_att_c) # [B,dunits]=[10,300]
                h2 = torch.tanh(preactx2)
                h2 = u2 * h1 + (1. - u2) * h2
                h2 = m_.unsqueeze(1) * h2 + (1. - m_).unsqueeze(1) * h1 # [B,dunits]=[10,300]            

                ## 1. RNN : s'(t)=RNN(y(t-1),s(t-1)) 
                # || to combine [emb(t-1),state(t-1)] -> predt-state(t) 
                # || RNN(dunits) : concat[ey,ctx-previous-posterior]
                preact3 = self.char2bpe_rnn1_U(hbpe_) # no bias -> [B,dunits*2]=[10,600]
                preact3 += c2b_x_ # [B,dunits*2]=[10,600]
                preact3 = torch.sigmoid(preact3)
                r3 = self._slice(preact3, 0, self.dunits) # reset gate [B,dunits]=[10,300]
                u3 = self._slice(preact3, 1, self.dunits) # update gate [B,dunits]=[10,300]
                preactx3 = self.char2bpe_rnn1_Ux(hbpe_) # no bias [B,dunits]=[10,300]
                preactx3 *= r3
                preactx3 += c2b_xx_ 
                h3 = torch.tanh(preactx3)
                h3 = u3 * hbpe_ + (1. - u3) * h3
                h3 = m_.unsqueeze(1) * h3 + (1. - m_).unsqueeze(1) * hbpe_

                ## 2. Attention : c(t)=AttentionContext(s'(t), enc_o) 
                # || dunits (prior_att_c[B,eprojs] prior_att_w[B,T])
                c2b_prior_att_c, c2b_prior_att_w = self.att_char2bpe[att_bpe_idx](cc_, hlens_bpe, h3, prev_c2b_alpha)
                c2b_prior_att_weights.append(c2b_prior_att_w)
                c2b_prior_att_context.append(c2b_prior_att_c)

                ## 3. RNN : s(t)=RNN(c(t),s'(t)) 
                # || att_c(eprojs) z_infocomb_list(dunits)
                h3 = torch.sigmoid( self.c2b_statecomb(h1) ) * h3 + h1
                preact4 = self.char2bpe_rnn2_U_n1(h3) # [B,dunits*2]=[10,600]
                preact4 += self.char2bpe_rnn2_Wc(c2b_prior_att_c) # no bias [B,dunits*2]=[10,600]
                preact4 = torch.sigmoid(preact4)
                r4 = self._slice(preact4, 0, self.dunits) # [B,dunits]
                u4 = self._slice(preact4, 1, self.dunits) # [B,dunits]
                preactx4 = self.char2bpe_rnn2_Ux_n1(h3) # [B,dunits]=[10,300]
                preactx4 *= r4
                preactx4 += self.char2bpe_rnn2_Wcx(c2b_prior_att_c) # [B,dunits]=[10,300]
                h4 = torch.tanh(preactx4)
                h4 = u4 * h3 + (1. - u4) * h4
                h4 = m_.unsqueeze(1) * h4 + (1. - m_).unsqueeze(1) * h3 # [B,dunits]=[10,300]

                ## Information combine for classification
                context = torch.cat([prior_att_c, c2b_prior_att_c],dim=1)
                cstates = torch.cat([h2, h4],dim=1)

                ## 4. Classification for every frames
                # h2[B,dunits]  state_bpe[B,decT]  prior_att_c[B,dunits//2]
                logit_all_for_classif = self.classbpe_state_W(cstates) + self.classbpe_yprev_W(state_bpe[:, i, :]) + self.classbpe_enc_W(context) # [B,dunits]=[10,300]
                logit_linear = torch.reshape( logit_all_for_classif, (batch, -1, 2) )
                logit_clf, _ = torch.max( logit_linear, dim=-1 ) # [B,dunits//2]=[B,150]
                probs = self.bpe_class_W(logit_clf) # [B,odim]=[10,496]
                probs = F.softmax(probs, dim=-1) # [B,496]

                if decode_choice == "bpe":
                    # ['▁A', '▁THE', '▁SE', '▁IM', '▁BUT', '▁M', '▁S', '▁IN', '▁B', '▁SAL']
                    max_len = int(max(char_len[i,:])) 
                    ysequence = self.eos_char * np.ones((nbatch, int(max_len+1) ), dtype=np.int) # (10,5)            
                    for subidx,chartoken in enumerate(yseq_char_inbpe[i]):
                        ysequence[subidx,:len(chartoken)] = np.array(chartoken)
                    ysequence = to_device(self, torch.from_numpy(ysequence)) # decT + 1
                    if switch_addcode_forbi == False: 
                        subprev_alpha, hbpe_, char_output_all, charposterindex = self.char_onebpe_training(
                            bpe_enc, hlens_bpe, subprev_alpha, hbpe_, ysequence, char_len[i,:], char_output_all, charposterindex) # char_len[i,:] is batch
                    else:
                        subprev_alpha, hbpe_, char_output_all, charposterindex = self.char_onebpe_training(
                            bpe_enc, hlens_bpe, subprev_alpha, hbpe_, ysequence, char_len[i,:], char_output_all, charposterindex, infor_bpe2char) # BIDIR                    

                ## 5. save posterior output
                bpe_output_all.append(probs)
                
                ## 6. update previous variables
                h_ = h2
                prev_alpha = prior_att_w
                prev_c2b_alpha = c2b_prior_att_w
            ###### Codes for BPE decoder and the additional operation (finish) ######

        ## 8. Get mainly loss : CE and auxiliary loss : KL-penalty
        ## 8.1 : char
        ypred_char = (char_output_all).view(batch * olength_char, self.odim_char) # [B,decT,odim_char]
        if traditional_loss_calculate == True:
            reduction_str = 'elementwise_mean' if LooseVersion(torch.__version__) < LooseVersion('1.0') else 'mean'
            # input : [N,C] label : [N] softmax is in F.cross_entropy (o:scalar)
            self.loss_char = F.categorical_crossentropy(ypred_char, ys_char_out_pad.view(-1), ignore_index=self.ignore_id, reduction=reduction_str)
            self.loss_char *= (np.mean([len(x) for x in ys_char_in]) - 1) # -1: eos, which is removed in the loss computation
        else:
            self.loss_char = F.categorical_crossentropy(ypred_char, ys_char_out_pad.contiguous().view(-1).long(), ignore_index=self.ignore_id, reduction='none')
            self.loss_char = self.loss_char.view(batch, olength_char) 
            self.loss_char = self.loss_char.sum(1)
            self.loss_char = self.loss_char.mean(0) # to be scalar
        ## 8.2 : bpe
        ypred_bpe = torch.stack(bpe_output_all, dim=1).view(batch * olength_bpe, self.odim_bpe)   # [B,decT,odim_bpe]
        if traditional_loss_calculate == True:
            reduction_str = 'elementwise_mean' if LooseVersion(torch.__version__) < LooseVersion('1.0') else 'mean'
            # input : [N,C] label : [N] softmax is in F.cross_entropy (o:scalar)
            self.loss_bpe = F.categorical_crossentropy(ypred_bpe, ys_bpe_out_pad.view(-1), ignore_index=self.ignore_id, reduction=reduction_str)
            self.loss_bpe *= (np.mean([len(x) for x in ys_bpe_in]) - 1) 
        else:
            self.loss_bpe = F.categorical_crossentropy(ypred_bpe, ys_bpe_out_pad.contiguous().view(-1).long(), ignore_index=self.ignore_id, reduction='none')
            self.loss_bpe = self.loss_bpe.view(batch, olength_bpe) 
            self.loss_bpe = self.loss_bpe.sum(1)
            self.loss_bpe = self.loss_bpe.mean(0) # to be scalar        

        # 9.1(CHAR) pad_outputs: prediction tensors (B*decoder_T, D)
        #           pad_targets: target tensors (B, decoder_T, D)
        acc_char = th_accuracy(ypred_char, ys_char_out_pad, ignore_label=self.ignore_id)
        logging.info('att_char loss:' + ''.join(str(self.loss_char.item()).split('\n')))
        # show predicted token sequence for debug (not changed it)
        if self.verbose > 0 and self.char_list is not None:
            ys_hat = ypred_char.view(batch, olength_char, -1)
            ys_true = ys_char_out_pad
            for (i, y_hat), y_true in zip(enumerate(ys_hat.detach().cpu().numpy()), ys_true.detach().cpu().numpy()):
                if i == MAX_DECODER_OUTPUT:
                    break
                idx_hat = np.argmax(y_hat[y_true != self.ignore_id], axis=1)
                idx_true = y_true[y_true != self.ignore_id]
                seq_hat = [self.char_list[int(idx)] for idx in idx_hat]
                seq_true = [self.char_list[int(idx)] for idx in idx_true]
                seq_hat = "".join(seq_hat)
                seq_true = "".join(seq_true)
                logging.info("groundtruth[%d]: " % i + seq_true)
                logging.info("prediction [%d]: " % i + seq_hat)
        # 9.2(BPE) pad_outputs: prediction tensors (B*decoder_T, D)
        #          pad_targets: target tensors (B, decoder_T, D)
        acc_bpe = th_accuracy(ypred_bpe, ys_bpe_out_pad, ignore_label=self.ignore_id)
        logging.info('att_bpe loss:' + ''.join(str(self.loss_bpe.item()).split('\n')))
        # show predicted token sequence for debug (not changed it)
        if self.verbose > 0 and self.bpe_list is not None:
            ys_hat = ypred_bpe.view(batch, olength_bpe, -1)
            ys_true = ys_bpe_out_pad
            for (i, y_hat), y_true in zip(enumerate(ys_hat.detach().cpu().numpy()), ys_true.detach().cpu().numpy()):
                if i == MAX_DECODER_OUTPUT:
                    break
                idx_hat = np.argmax(y_hat[y_true != self.ignore_id], axis=1)
                idx_true = y_true[y_true != self.ignore_id]
                seq_hat = [self.bpe_list[int(idx)] for idx in idx_hat]
                seq_true = [self.bpe_list[int(idx)] for idx in idx_true]
                seq_hat = "".join(seq_hat)
                seq_true = "".join(seq_true)
                logging.info("groundtruth[%d]: " % i + seq_true)
                logging.info("prediction [%d]: " % i + seq_hat)    

        ## 10.1 Label smoothing char 
        if self.labeldist_char is not None: 
            if self.vlabeldist_char is None:
                self.vlabeldist_char = to_device(self, torch.from_numpy(self.labeldist_char))
            loss_reg_char = - torch.sum((torch.log(ypred_char) * self.vlabeldist_char).view(-1), dim=0) / len(ys_char_in)
            self.loss_char = (1. - self.lsm_weight) * self.loss_char + self.lsm_weight * loss_reg_char
        ## 10.2 Label smoothing bpe
        if self.labeldist_bpe is not None: 
            if self.vlabeldist_bpe is None:
                self.vlabeldist_bpe = to_device(self, torch.from_numpy(self.labeldist_bpe))
            loss_reg_bpe = - torch.sum((torch.log(ypred_bpe) * self.vlabeldist_bpe).view(-1), dim=0) / len(ys_bpe_in)
            self.loss_bpe = (1. - self.lsm_weight) * self.loss_bpe + self.lsm_weight * loss_reg_bpe

        return [self.loss_char, self.loss_bpe], [acc_char, acc_bpe]



    # Not Finished (not check). this code perherps useless
    def recognize_beam(self, h, lpz, recog_args, list, rnnlm=None, strm_idx=0): # h(enc_o) [T,D_enc] lpz(CTC output)
        ## empty
        return 0 # remove sos



    def char_onebpe_training(self, exp_h, exp_hlens, prev_alpha, h_, yseq, char_len, char_output_all, charposterindex, bpe2char_state=None, strm_idx=0):  # realmask
        # exp_h         ：torch.Tensor (beam,decT,dunit)
        # exp_hlens     ：list
        # previous_alpha：torch.Tensor (beam,decT)
        # previous_state：torch.Tensor (beam,dunit)
        # yseq          ：numpy.ndarray
        # char_len      ：numpy.ndarray
        att_char_idx = min(strm_idx, len(self.att_char) - 1)
        realmask = (to_device(self, make_trueframe_mask(char_len))).float()       # (1 is useful)
        nbatch = exp_h.shape[0]
        output_all = torch.zeros((int(max(char_len)), nbatch, self.odim_char)).to(exp_h) # (decT,batch,odim)

        for ii in range(int(max(char_len))):
            vy = yseq[:,ii] # to_device(self, torch.LongTensor((yseq[:,i])))         # get the last char
            ey = self.dropout_emb_char(self.embed_char(vy)) # [nbatch, embsize]
            xx_ = self.char_rnn1_Wx(ey) # state_belowx[nbatch, dunits]
            x_ = self.char_rnn1_W(ey) # [nbatch, 2*dunits]
            cc_ = exp_h # [nbatch, T, eprojs]
            m_ = realmask[:,ii] # control Mask

            ## 1. RNN eqn.(22): s'(t)=RNN(y(t-1),s(t-1)) || to combine [emb(t-1),state(t-1)] -> predt-state(t) || RNN(dunits) : concat[ey,ctx-previous-posterior]
            preact1 = self.char_rnn1_U(h_) # [nbatch,dunits*2]
            preact1 += x_ # [nbatch,dunits*2]
            preact1 = torch.sigmoid(preact1)
            r1 = self._slice(preact1, 0, self.dunits) # reset gate [nbatch,dunits]
            u1 = self._slice(preact1, 1, self.dunits) # update gate [nbatch,dunits]
            preactx1 = self.char_rnn1_Ux(h_) # [nbatch,dunits]
            preactx1 *= r1
            preactx1 += xx_ 
            h1 = torch.tanh(preactx1)
            h1 = u1 * h_ + (1. - u1) * h1
            h1 = m_.unsqueeze(1) * h1 + (1. - m_).unsqueeze(1) * h_

            ## 2. Attention eqn.(23): c(t)=AttentionContext(s'(t), enc_o) || dunits
            prior_att_c, prior_att_w = self.att_char[att_char_idx](cc_, exp_hlens, h1, prev_alpha)

            ## 3. RNN eqn.(24): s(t)=RNN(c(t),s'(t)) || att_c(eprojs) z_infocomb_list(dunits)
            preact2 = self.char_rnn2_U_n1(h1) # [nbatch,dunits*2]
            preact2 += self.char_rnn2_Wc(prior_att_c) # [nbatch,dunits*2]
            preact2 = torch.sigmoid(preact2)
            r2 = self._slice(preact2, 0, self.dunits) # [nbatch,dunits]
            u2 = self._slice(preact2, 1, self.dunits) # [nbatch,dunits]            
            preactx2 = self.char_rnn2_Ux_n1(h1) # [nbatch,dunits]
            preactx2 *= r2
            preactx2 += self.char_rnn2_Wcx(prior_att_c) # [nbatch,dunits]
            h2 = torch.tanh(preactx2)
            h2 = u2 * h1 + (1. - u2) * h2 
            h2 = m_.unsqueeze(1) * h2 + (1. - m_).unsqueeze(1) * h1 # [nbatch,dunits]

            ## 4. Do classification for every frame
            if switch_addcode_forbi == False:
                logit_all_for_classif = self.classchar_state_W(h2) + self.classchar_yprev_W(ey) + self.classchar_enc_W(prior_att_c) # [nbatch,dunits]
            else:
                shortline = self.classchar_state_W(h2) + self.classchar_yprev_W(ey) + self.classchar_enc_W(prior_att_c)
                branch_pass = torch.relu( ( self.classb2c_state_W1(shortline) ) )
                branch_pass =  self.classb2c_state_W2(branch_pass) 
                logit_all_for_classif = shortline + branch_pass # [nbatch,dunits]

            ## ## ## 
            similarb = logit_all_for_classif.size(0)                                                                            # 
            logit_linear    = torch.reshape( logit_all_for_classif, (similarb, self.dunits//2, 2) )
            logit_clf, _ = torch.max( logit_linear, dim=2 )                                                                     # [nbatch,dunits//2]
            probs = self.char_class_W(logit_clf)                                                                                # [nbatch,odim]
            probs = F.softmax(probs, dim=1)                                                                                     # [nbatch,odim]

            ## 5. Update the alpha and the posterior output
            prev_alpha = m_.unsqueeze(1) * prior_att_w + (1. - m_).unsqueeze(1) * prev_alpha                                    # [nbatch,dunits]
            h_ = h2
            output_all[ii,:,:] = probs # [B,odim] output_all(decT,B,odim)

        # Concate all the posterior output.
        for bidx in range(nbatch):
            out_idx = charposterindex[bidx]
            char_output_all[bidx,out_idx:out_idx+char_len[bidx],:] = output_all[0:char_len[bidx],bidx,:]
        # Then update the posterior index
        charposterindex += char_len

        return prev_alpha, h_, char_output_all, charposterindex



    def calculate_all_attentions(self, hs_pad, hlens, ys_pad_char, ys_pad_bpe, strm_idx=0):  #### change here
        """Decoder forward

            :param torch.Tensor hs_pad: batch of padded hidden state sequences (B, Tmax, D) : enc_o
            :param torch.Tensor hlens: batch of lengths of hidden state sequences (B) : enc_o_len
            :param torch.Tensor ys_pad1: batch of padded token id sequence tensor (B, Lmax1) : char true_label
            :param torch.Tensor ys_pad1: batch of padded token id sequence tensor (B, Lmax2) : bpe true_label
            :param int strm_idx: stream index indicates the index of decoding stream. (y_idx ? )
            :return: attention loss value
            :rtype: torch.Tensor
            :return: accuracy
            :rtype: float
        """
        SWITCH_BPE = True
        SWITCH_CHAR = True
        nbatch = hs_pad.shape[0]

        if decode_choice == "bpe":
            ###### 0.Codes for CHAR decoder and the additional operation (start)  ######
            # initialization : give initial context_vector & decoder_state
            
            # TODO(kan-bayashi): need to make more smart way
            ys_char = [y[y != self.ignore_id] for y in ys_pad_char]  # parse padded ys (char) List

            # attention index for the attention module in SPA (speaker parallel attention), att_idx is used to select attention module. In other cases, it is 0.
            att_char_idx = min(strm_idx, len(self.att_char) - 1) # useful only for SAT

            # reset pre-computation of h
            self.att_char[att_char_idx].reset()

            # prepare input and output word sequences with sos/eos IDs (char)
            eos_char = ys_char[0].new([self.eos_char])
            sos_char = ys_char[0].new([self.sos_char])
            ys_char_in = [torch.cat([sos_char, y], dim=0) for y in ys_char] # in this way : the label and the feedback y is created (ys_in is for state(t-1))
            ys_char_out = [torch.cat([y, eos_char], dim=0) for y in ys_char] #  

            # char y-label length
            ys_char_out_len = [y.size(0) for y in ys_char_out]
            y_char_mask = (to_device(self, make_trueframe_mask(ys_char_out_len))).float()

            # input mask
            context_mask = (to_device(self, make_trueframe_mask(hlens))).float() # [B,T]=[10,203] torch.Tensor

            # padding for ys with -1
            # pys: utt x olen (char)
            ys_char_in_pad  = pad_list(ys_char_in, self.eos_char).to(hs_pad).long() # pad feature torch.Tensor([B,decT])
            ys_char_out_pad = pad_list(ys_char_out, self.ignore_id).to(hs_pad).long() # pad output symbol

            # pre-computation of embedding y -> emb(y) [B,C,D_senone]
            eys_char = self.dropout_emb_char(self.embed_char(ys_char_in_pad))  # utt x olen x zdim [B,decoder_T,300]

            # hlen should be list of integer
            hlens_char = list(map(int, hlens))

            # get dim, length info [B,T_dec] = [15,168]
            olength_char = ys_char_out_pad.size(1)
            logging.info(self.__class__.__name__ + ' input(speech) lengths:  ' + str(hlens_char))
            logging.info(self.__class__.__name__ + ' output(char) lengths: ' + str([y.size(0) for y in ys_char_out]))

            # Unchanged varibles
            char_enc = hs_pad
            subh_ = self.mean_init_char_state(char_enc, hlens_char)                         #.repeat(n_bb, 1) # [n_bb, dunits]
            subprev_energy = (to_device(self, make_trueframe_mask(hlens_char))).float()     # previous postr attention score
            subprev_alpha = ( subprev_energy / (subprev_energy.sum(1, keepdim=True)+1e-8) ) # [n_bb, ]           

            char_prior_att_weights = []
            char_prior_att_context = []

            # This is for the char decoder process
            state_char  = eys_char                   # y(t-1): [B,decT,embsize_char]=[B,144,256]
            state_char_ = self.char_rnn1_W(eys_char)  # [B,decT,dunits]=[10,144,600]
            state_charx = self.char_rnn1_Wx(eys_char) # [B,decT,dunits]=[10,144,300]
            ###### 0.Codes for CHAR decoder and the additional operation (finish) ######       

        if SWITCH_BPE == True:
            ###### Codes for BPE decoder and the additional operation (start)  ######
            # initialization : give initial context_vector & decoder_state
            
            # TODO(kan-bayashi): need to make more smart way
            ys_bpe = [y[y != self.ignore_id] for y in ys_pad_bpe]  # parse padded ys (bpe) List

            # attention index for the attention module in SPA (speaker parallel attention), att_idx is used to select attention module. In other cases, it is 0.
            att_bpe_idx = min(strm_idx, len(self.att_bpe) - 1) # useful only for SAT

            # reset pre-computation of h
            self.att_bpe[att_bpe_idx].reset()
            self.att_char2bpe[att_bpe_idx].reset()

            # prepare input and output word sequences with sos/eos IDs (bpe)
            eos_bpe = ys_bpe[0].new([self.eos_bpe])
            sos_bpe = ys_bpe[0].new([self.sos_bpe])
            ys_bpe_in = [torch.cat([sos_bpe, y], dim=0) for y in ys_bpe] # in this way : the label and the feedback y is created (ys_in is for state(t-1))
            ys_bpe_out = [torch.cat([y, eos_bpe], dim=0) for y in ys_bpe] # 

            ys_bpe_all = (pad_list(ys_bpe_out, self.ignore_id)).transpose(1, 0).contiguous().tolist() # Torch.tensor((batch,decT)) : -1 means useless
            decTlen = len(ys_bpe_all)

            ysbpe_subseqs = [[] for _ in range(decTlen)]
            char_len = np.zeros((decTlen, nbatch), dtype=np.int)

            yseq_char_inbpe = [[] for _ in range(decTlen)]
            yseq_char_inbpe[0] = [[self.sos_char] for y in six.moves.range(nbatch)] ## 只是为了循环体的变量

            for trans_idx, yseqs in enumerate(ys_bpe_all): # yseqs list(batch)  (decT(*),B)
                bpe_token_list = [self.bpedict[idx] for idx in yseqs] # ['▁ALSO']

                for idx,bpe_token in enumerate(bpe_token_list):
                    if ((trans_idx == 0) and bpe_token == "▁") or (bpe_token == "<padding>"): ## 如果第一个字符是"▁", 那么：就选择忘记这个BPE的情况continue:
                        char_len[trans_idx,idx] = len(yseq_char_inbpe[trans_idx][idx]) - 1
                        continue                    
                    elif (trans_idx == 0) and (bpe_token[0] == "▁"):
                        token = bpe_token[1:]
                    else:
                        token = bpe_token
                   
                    if token == '[laughter]' or token == '[noise]' or token == '[vocalized-noise]' or token == '<eos>' or token == '<blank>' or token == '<unk>': #   <unk> <blank> <eos>
                        yseq_char_inbpe[trans_idx][idx].append(self.inverse_chardict[token]) ## all length
                    else:
                        for iidx, str_ in enumerate(token):
                            yseq_char_inbpe[trans_idx][idx].append(self.inverse_chardict[str_]) ## all length

                    char_len[trans_idx,idx] = len(yseq_char_inbpe[trans_idx][idx]) - 1

                if trans_idx+1 < decTlen: yseq_char_inbpe[trans_idx+1] = [[chartoken[-1]] for chartoken in yseq_char_inbpe[trans_idx]]

            # bpe y-label length
            ys_bpe_out_len = [y.size(0) for y in ys_bpe_out]
            y_bpe_mask = (to_device(self, make_trueframe_mask(ys_bpe_out_len))).float()

            # input mask
            context_mask = (to_device(self, make_trueframe_mask(hlens))).float() # [B,T]=[10,203] torch.Tensor

            # padding for ys with -1
            # pys: utt x olen (bpe)
            ys_bpe_in_pad  = pad_list(ys_bpe_in, self.eos_bpe).to(hs_pad).long() # pad feature torch.Tensor([B,decT])
            ys_bpe_out_pad = pad_list(ys_bpe_out, self.ignore_id).to(hs_pad).long() # pad output symbol

            # pre-computation of embedding y -> emb(y) [B,C,D_senone]
            eys_bpe = self.dropout_emb_bpe(self.embed_bpe(ys_bpe_in_pad))  # utt x olen x zdim [B,decoder_T,300]

            # hlen should be list of integer
            hlens_bpe = list(map(int, hlens))

            # get dim, length info [B,T_dec] = [15,168]
            batch = ys_bpe_out_pad.size(0)
            olength_bpe = ys_bpe_out_pad.size(1)
            logging.info(self.__class__.__name__ + ' input(speech) lengths:  ' + str(hlens_bpe))
            logging.info(self.__class__.__name__ + ' output(bpe) lengths: ' + str([y.size(0) for y in ys_bpe_out]))

            # Unchanged varibles
            bpe_enc = hs_pad
            h_ = self.mean_init_bpe_state(bpe_enc, hlens_bpe) # 
            hbpe_ = self.mean_init_bpe_state(bpe_enc, hlens_bpe) # hlens_bpe ?
            prev_postr_energy = (to_device(self, make_trueframe_mask(hlens_bpe))).float() # previous postr attention score
            prev_alpha = prev_postr_energy / (prev_postr_energy.sum(1, keepdim=True)+1e-8) # [B,T]      
            prev_c2b_alpha = prev_alpha.clone()      
            
            bpe_prior_att_weights = []
            bpe_prior_att_context = []
            bpe_output_all = []       
            c2b_prior_att_weights = []
            c2b_prior_att_context = []             

            # This is for the bpe decoder process
            state_bpe  = eys_bpe                   # y(t-1): [B,decT,embsize_bpe]=[B,144,256]
            state_bpe_ = self.bpe_rnn1_W(eys_bpe)  # [B,decT,dunits]=[10,144,600]
            state_bpex = self.bpe_rnn1_Wx(eys_bpe) # [B,decT,dunits]=[10,144,300]
            charposterindex = np.zeros(nbatch, dtype=np.int)
            char_output_all = (1.0/self.odim_char) * torch.ones((nbatch, olength_char, self.odim_char)).to(hs_pad)

            state_char2bpe_ = self.char2bpe_rnn1_W(eys_bpe)  # [B,decT,dunits]=[10,144,600]
            state_char2bpex = self.char2bpe_rnn1_Wx(eys_bpe) # [B,decT,dunits]=[10,144,300]
            for i in six.moves.range(olength_bpe):
                m_  = y_bpe_mask[:,i] # the variables for this bpe decoder steps
                x_  = state_bpe_[:,i,:]
                xx_ = state_bpex[:,i,:]
                cc_ = bpe_enc # [B,decT,dunit//2]
                c2b_x_  = state_char2bpe_[:,i,:]
                c2b_xx_ = state_char2bpex[:,i,:]                

                ## 1. RNN : s'(t)=RNN(y(t-1),s(t-1)) 
                # || to combine [emb(t-1),state(t-1)] -> predt-state(t) 
                # || RNN(dunits) : concat[ey,ctx-previous-posterior]
                if switch_addcode_forbi == True: infor_bpe2char = h_
                preact1 = self.bpe_rnn1_U(h_) # no bias -> [B,dunits*2]=[10,600]
                preact1 += x_ # [B,dunits*2]=[10,600]
                preact1 = torch.sigmoid(preact1)
                r1 = self._slice(preact1, 0, self.dunits) # reset gate [B,dunits]=[10,300]
                u1 = self._slice(preact1, 1, self.dunits) # update gate [B,dunits]=[10,300]
                preactx1 = self.bpe_rnn1_Ux(h_) # no bias [B,dunits]=[10,300]
                preactx1 *= r1
                preactx1 += xx_ 
                h1 = torch.tanh(preactx1)
                h1 = u1 * h_ + (1. - u1) * h1
                h1 = m_.unsqueeze(1) * h1 + (1. - m_).unsqueeze(1) * h_

                ## 2. Attention : c(t)=AttentionContext(s'(t), enc_o) 
                # || dunits (prior_att_c[B,eprojs] prior_att_w[B,T])
                prior_att_c, prior_att_w = self.att_bpe[att_bpe_idx](cc_, hlens_bpe, h1, prev_alpha)
                bpe_prior_att_weights.append(prior_att_w)
                bpe_prior_att_context.append(prior_att_c)

                ## 3. RNN : s(t)=RNN(c(t),s'(t)) 
                # || att_c(eprojs) z_infocomb_list(dunits)
                preact2 = self.bpe_rnn2_U_n1(h1) # [B,dunits*2]=[10,600]
                preact2 += self.bpe_rnn2_Wc(prior_att_c) # no bias [B,dunits*2]=[10,600]
                preact2 = torch.sigmoid(preact2)
                r2 = self._slice(preact2, 0, self.dunits) # [B,dunits]
                u2 = self._slice(preact2, 1, self.dunits) # [B,dunits]
                preactx2 = self.bpe_rnn2_Ux_n1(h1) # [B,dunits]=[10,300]
                preactx2 *= r2
                preactx2 += self.bpe_rnn2_Wcx(prior_att_c) # [B,dunits]=[10,300]
                h2 = torch.tanh(preactx2)
                h2 = u2 * h1 + (1. - u2) * h2
                h2 = m_.unsqueeze(1) * h2 + (1. - m_).unsqueeze(1) * h1 # [B,dunits]=[10,300]            

                ## 1. RNN : s'(t)=RNN(y(t-1),s(t-1)) 
                # || to combine [emb(t-1),state(t-1)] -> predt-state(t) 
                # || RNN(dunits) : concat[ey,ctx-previous-posterior]
                preact3 = self.char2bpe_rnn1_U(hbpe_) # no bias -> [B,dunits*2]=[10,600]
                preact3 += c2b_x_ # [B,dunits*2]=[10,600]
                preact3 = torch.sigmoid(preact3)
                r3 = self._slice(preact3, 0, self.dunits) # reset gate [B,dunits]=[10,300]
                u3 = self._slice(preact3, 1, self.dunits) # update gate [B,dunits]=[10,300]
                preactx3 = self.char2bpe_rnn1_Ux(hbpe_) # no bias [B,dunits]=[10,300]
                preactx3 *= r3
                preactx3 += c2b_xx_ 
                h3 = torch.tanh(preactx3)
                h3 = u3 * hbpe_ + (1. - u3) * h3
                h3 = m_.unsqueeze(1) * h3 + (1. - m_).unsqueeze(1) * hbpe_

                ## 2. Attention : c(t)=AttentionContext(s'(t), enc_o) 
                # || dunits (prior_att_c[B,eprojs] prior_att_w[B,T])
                c2b_prior_att_c, c2b_prior_att_w = self.att_char2bpe[att_bpe_idx](cc_, hlens_bpe, h3, prev_c2b_alpha)
                c2b_prior_att_weights.append(c2b_prior_att_w)
                c2b_prior_att_context.append(c2b_prior_att_c)

                ## 3. RNN : s(t)=RNN(c(t),s'(t)) 
                # || att_c(eprojs) z_infocomb_list(dunits)
                h3 = torch.sigmoid( self.c2b_statecomb(h1) ) * h3 + h1
                preact4 = self.char2bpe_rnn2_U_n1(h3) # [B,dunits*2]=[10,600]
                preact4 += self.char2bpe_rnn2_Wc(c2b_prior_att_c) # no bias [B,dunits*2]=[10,600]
                preact4 = torch.sigmoid(preact4)
                r4 = self._slice(preact4, 0, self.dunits) # [B,dunits]
                u4 = self._slice(preact4, 1, self.dunits) # [B,dunits]
                preactx4 = self.char2bpe_rnn2_Ux_n1(h3) # [B,dunits]=[10,300]
                preactx4 *= r4
                preactx4 += self.char2bpe_rnn2_Wcx(c2b_prior_att_c) # [B,dunits]=[10,300]
                h4 = torch.tanh(preactx4)
                h4 = u4 * h3 + (1. - u4) * h4
                h4 = m_.unsqueeze(1) * h4 + (1. - m_).unsqueeze(1) * h3 # [B,dunits]=[10,300]

                ## Information combine for classification
                context = torch.cat([prior_att_c, c2b_prior_att_c],dim=1)
                cstates = torch.cat([h2, h4],dim=1)

                ## 4. Classification for every frames
                # h2[B,dunits]  state_bpe[B,decT]  prior_att_c[B,dunits//2]
                logit_all_for_classif = self.classbpe_state_W(cstates) + self.classbpe_yprev_W(state_bpe[:, i, :]) + self.classbpe_enc_W(context) # [B,dunits]=[10,300]
                logit_linear = torch.reshape( logit_all_for_classif, (batch, -1, 2) )
                logit_clf, _ = torch.max( logit_linear, dim=-1 ) # [B,dunits//2]=[B,150]
                probs = self.bpe_class_W(logit_clf) # [B,odim]=[10,496]
                probs = F.softmax(probs, dim=-1) # [B,496]

                if decode_choice == "bpe":
                    # ['▁A', '▁THE', '▁SE', '▁IM', '▁BUT', '▁M', '▁S', '▁IN', '▁B', '▁SAL']
                    max_len = int(max(char_len[i,:])) 
                    ysequence = self.eos_char * np.ones((nbatch, int(max_len+1) ), dtype=np.int) # (10,5)            
                    for subidx,chartoken in enumerate(yseq_char_inbpe[i]):
                        ysequence[subidx,:len(chartoken)] = np.array(chartoken)
                    ysequence = to_device(self, torch.from_numpy(ysequence)) # decT + 1

                    if switch_addcode_forbi == False: 
                        subprev_alpha, hbpe_, char_output_all, charposterindex = self.char_onebpe_training(
                            bpe_enc, hlens_bpe, subprev_alpha, hbpe_, ysequence, char_len[i,:], char_output_all, charposterindex) # char_len[i,:] is batch
                    else:
                        subprev_alpha, hbpe_, char_output_all, charposterindex = self.char_onebpe_training(
                            bpe_enc, hlens_bpe, subprev_alpha, hbpe_, ysequence, char_len[i,:], char_output_all, charposterindex, infor_bpe2char) # char_len[i,:] is batch

                ## 5. save posterior output
                bpe_output_all.append(probs)
                
                ## 6. update previous variables
                h_ = h2
                prev_alpha = prior_att_w
                prev_c2b_alpha = c2b_prior_att_w
            ###### Codes for BPE decoder and the additional operation (finish) ######
        
        att_ws_bpe = att_to_numpy(bpe_prior_att_weights, self.att_bpe[att_bpe_idx]) 
        att_ws_c2b = att_to_numpy(c2b_prior_att_weights, self.att_bpe[att_bpe_idx]) 
        return (att_ws_bpe, att_ws_c2b)


    
    def recognize_beam_batch(self, h, hlens, lpz, recog_args, char_list, bpe_list=None, rnnlm=None, normalize_score=True, strm_idx=0):
        logging.info('input lengths: ' + str(h.size(1)))                                                        # [B,T,feadim]
        att_char_idx = min(strm_idx, len(self.att_char) - 1)                                                    # speaker adapation choice
        att_bpe_idx = min(strm_idx, len(self.att_bpe) - 1)                                                      # speaker adapation choice
        if decode_choice == 'char': 
            odim_nums = self.odim_char 
        elif decode_choice == 'bpe': 
            odim_nums = self.odim_bpe

        h = mask_by_length(h, hlens, 0.0)                                                                       # mask feature

        # search params
        CHOICE_SCORE_TYPES = recog_args.CHOICE_SCORE_TYPES
        batch = len(hlens)                                                                                      # 
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = recog_args.ctc_weight
        att_weight = 1.0 - ctc_weight

        n_bb = batch * beam                                                                                     # the true batch input (the beamsearch intermediate size)
        n_bo = beam * odim_nums                                                                                 # the true batch senone
        n_bbo = n_bb * odim_nums
        pad_b = to_device(self, torch.LongTensor([i * beam for i in six.moves.range(batch)]).view(-1, 1))       # stable
        pad_bo = to_device(self, torch.LongTensor([i * n_bo for i in six.moves.range(batch)]).view(-1, 1))      # 
        pad_o = to_device(self, torch.LongTensor([i * odim_nums for i in six.moves.range(n_bb)]).view(-1, 1))   # 

        max_hlen = int(max(hlens))
        if recog_args.maxlenratio == 0:
            maxlen = max_hlen
        else:
            maxlen = max(1, int(recog_args.maxlenratio * max_hlen))
        minlen = int(recog_args.minlenratio * max_hlen)
        logging.info('max output length: ' + str(maxlen))
        logging.info('min output length: ' + str(minlen))                                                       # stable

        # initialization
        vscores = to_device(self, torch.zeros(batch, beam))                                                     # [B, Beam] final score
        if (decode_choice == 'bpe'):
            vscores_char = to_device(self, torch.zeros(batch, beam))

        rnnlm_prev = None

        if decode_choice == 'char':
            self.att_char[att_char_idx].reset()  # reset pre-computation of h
            sos_token, eos_token = self.sos_char, self.eos_char
        elif decode_choice == 'bpe':
            self.att_bpe[att_bpe_idx].reset()
            self.att_char2bpe[att_bpe_idx].reset()
            sos_token, eos_token = self.sos_bpe, self.eos_bpe
            self.att_char[att_char_idx].reset()

        yseq = [[sos_token] for _ in six.moves.range(n_bb)]
        accum_odim_ids = [sos_token for _ in six.moves.range(n_bb)]

        stop_search = [False for _ in six.moves.range(batch)]
        nbest_hyps = [[] for _ in six.moves.range(batch)]
        ended_hyps = [[] for _ in range(batch)]

        exp_hlens = hlens.repeat(beam).view(beam, batch).transpose(0, 1).contiguous()        
        exp_hlens = exp_hlens.view(-1).tolist()
        exp_h = h.unsqueeze(1).repeat(1, beam, 1, 1).contiguous()
        exp_h = exp_h.view(n_bb, h.size()[1], h.size()[2]) # [n_bb, T, eprojs]

        if (decode_choice == 'char'):
            # extra initialization        
            h_ = self.mean_init_char_state(exp_h, list(map(int, exp_hlens))) #.repeat(n_bb, 1) # [n_bb, dunits]
            prev_postr_energy = (to_device(self, make_trueframe_mask(exp_hlens))).float() # previous postr attention score
            prev_alpha = ( prev_postr_energy / (prev_postr_energy.sum(1, keepdim=True)+1e-8) ) # [n_bb, ]
        elif (decode_choice == 'bpe'):
            # extra initialization        
            h_ = self.mean_init_bpe_state(exp_h, list(map(int, exp_hlens))) #.repeat(n_bb, 1) # [n_bb, dunits]
            hbpe_ = self.mean_init_bpe_state(exp_h, list(map(int, exp_hlens))) #.repeat(n_bb, 1) # [n_bb, dunits]
            prev_postr_energy = (to_device(self, make_trueframe_mask(exp_hlens))).float() # previous postr attention score
            prev_alpha = ( prev_postr_energy / (prev_postr_energy.sum(1, keepdim=True)+1e-8) ) # [n_bb, ]
            prev_c2b_alpha = prev_alpha.clone()      
            # for char sequence
            yseq_char = [[self.sos_char] for y in six.moves.range(n_bb)]
            subh_ = self.mean_init_char_state(exp_h, list(map(int, exp_hlens)))                                     #.repeat(n_bb, 1) # [n_bb, dunits]
            subprev_energy = (to_device(self, make_trueframe_mask(exp_hlens))).float()                              # previous postr attention score
            subprev_alpha = ( subprev_energy / (subprev_energy.sum(1, keepdim=True)+1e-8) )                         # [n_bb, ]           
            yseq_char_inbpe = [[self.sos_char] for y in six.moves.range(n_bb)]

        if lpz is not None: # no need to modification
            device_id = torch.cuda.device_of(next(self.parameters()).data).idx
            ctc_prefix_score = CTCPrefixScoreTH(lpz, 0, eos_token, beam, exp_hlens, device_id)
            ctc_states_prev = ctc_prefix_score.initial_state()
            ctc_scores_prev = to_device(self, torch.zeros(batch, n_bo))

        char_prior_att_weights = []
        char_prior_att_context = []
        bpe_prior_att_weights = []
        bpe_prior_att_context = []        
        output_all = []

        for i in six.moves.range(maxlen):
            if decode_choice == 'char':
                logging.debug('CHAR position ' + str(i))
                vy = to_device(self, torch.LongTensor(get_last_yseq(yseq))) # get the last char
                ey = self.dropout_emb_char(self.embed_char(vy)) # [n_bb, embsize]
                xx_ = self.char_rnn1_Wx(ey) # state_belowx[n_bb, dunits]
                x_ = self.char_rnn1_W(ey) # [n_bb, 2*dunits]
                cc_ = exp_h # [n_bb, T, eprojs]

                ## 1. RNN eqn.(22): s'(t)=RNN(y(t-1),s(t-1)) || to combine [emb(t-1),state(t-1)] -> predt-state(t) || RNN(dunits) : concat[ey,ctx-previous-posterior]
                preact1 = self.char_rnn1_U(h_) # [n_bb,dunits*2]
                preact1 += x_ # [n_bb,dunits*2]
                preact1 = torch.sigmoid(preact1)
                r1 = self._slice(preact1, 0, self.dunits) # reset gate [n_bb,dunits]
                u1 = self._slice(preact1, 1, self.dunits) # update gate [n_bb,dunits]
                preactx1 = self.char_rnn1_Ux(h_) # [n_bb,dunits]
                preactx1 *= r1
                preactx1 += xx_ 
                h1 = torch.tanh(preactx1)
                h1 = u1 * h_ + (1. - u1) * h1
                # h1 = m_.unsqueeze(1) * h1 + (1. - m_).unsqueeze(1) * h_

                ## 2. Attention eqn.(23): c(t)=AttentionContext(s'(t), enc_o) || dunits
                prior_att_c, prior_att_w = self.att_char[att_char_idx](cc_, exp_hlens, h1, prev_alpha)
                char_prior_att_weights.append(prior_att_w)
                char_prior_att_context.append(prior_att_c)   

                ## 3. RNN eqn.(24): s(t)=RNN(c(t),s'(t)) || att_c(eprojs) z_infocomb_list(dunits)
                preact2 = self.char_rnn2_U_n1(h1) # [n_bb,dunits*2]
                preact2 += self.char_rnn2_Wc(prior_att_c) # [n_bb,dunits*2]
                preact2 = torch.sigmoid(preact2)
                r2 = self._slice(preact2, 0, self.dunits) # [n_bb,dunits]
                u2 = self._slice(preact2, 1, self.dunits) # [n_bb,dunits]            
                preactx2 = self.char_rnn2_Ux_n1(h1) # [n_bb,dunits]
                preactx2 *= r2
                preactx2 += self.char_rnn2_Wcx(prior_att_c) # [n_bb,dunits]
                h2 = torch.tanh(preactx2)
                h2 = u2 * h1 + (1. - u2) * h2 
                # h2 = m_.unsqueeze(1) * h2 + (1. - m_).unsqueeze(1) * h1 # [n_bb,dunits]

                ## 4. Do classification for every frame
                logit_all_for_classif = self.classchar_state_W(h2) + self.classchar_yprev_W(ey) + self.classchar_enc_W(prior_att_c) # [n_bb,dunits]
                similarb = logit_all_for_classif.size(0) # 
                logit_linear    = torch.reshape( logit_all_for_classif, (similarb, self.dunits//2, 2) )
                logit_clf, _ = torch.max( logit_linear, dim=2 ) # [n_bb,dunits//2]
                probs = self.char_class_W(logit_clf) # [n_bb,odim]
                probs = F.softmax(probs, dim=1) # [n_bb,odim]

            elif decode_choice == 'bpe':
                logging.debug('Mixed(mltaskr) BPE position ' + str(i))
                vy = to_device(self, torch.LongTensor(get_last_yseq(yseq))) # get the last bpe
                ey = self.dropout_emb_bpe(self.embed_bpe(vy)) # [n_bb, embsize]
                xx_ = self.bpe_rnn1_Wx(ey) # state_belowx[n_bb, dunits]
                x_ = self.bpe_rnn1_W(ey) # [n_bb, 2*dunits]
                c2b_xx_ = self.char2bpe_rnn1_Wx(ey) # state_belowx[n_bb, dunits]
                c2b_x_  = self.char2bpe_rnn1_W(ey)                
                cc_ = exp_h # [n_bb, T, eprojs]

                ## 1. RNN eqn.(22): s'(t)=RNN(y(t-1),s(t-1)) 
                # || to combine [emb(t-1),state(t-1)] -> predt-state(t) 
                # || RNN(dunits) : concat[ey,ctx-previous-posterior]
                if switch_addcode_forbi == False: infor_bpe2char = h_
                preact1 = self.bpe_rnn1_U(h_) # [n_bb,dunits*2]
                preact1 += x_ # [n_bb,dunits*2]
                preact1 = torch.sigmoid(preact1)
                r1 = self._slice(preact1, 0, self.dunits) # reset gate [n_bb,dunits]
                u1 = self._slice(preact1, 1, self.dunits) # update gate [n_bb,dunits]
                preactx1 = self.bpe_rnn1_Ux(h_) # [n_bb,dunits]
                preactx1 *= r1
                preactx1 += xx_ 
                h1 = torch.tanh(preactx1)
                h1 = u1 * h_ + (1. - u1) * h1
                # h1 = m_.unsqueeze(1) * h1 + (1. - m_).unsqueeze(1) * h_

                ## 2. Attention eqn.(23): c(t)=AttentionContext(s'(t), enc_o) 
                # || context vectot / attention weight : dunits
                prior_att_c, prior_att_w = self.att_bpe[att_bpe_idx](cc_, exp_hlens, h1, prev_alpha)

                ## 3. RNN eqn.(24): s(t)=RNN(c(t),s'(t)) 
                # || att_c(eprojs) z_infocomb_list(dunits)
                preact2 = self.bpe_rnn2_U_n1(h1) # [n_bb,dunits*2]
                preact2 += self.bpe_rnn2_Wc(prior_att_c) # [n_bb,dunits*2]
                preact2 = torch.sigmoid(preact2)
                r2 = self._slice(preact2, 0, self.dunits) # [n_bb,dunits]
                u2 = self._slice(preact2, 1, self.dunits) # [n_bb,dunits]            
                preactx2 = self.bpe_rnn2_Ux_n1(h1) # [n_bb,dunits]
                preactx2 *= r2
                preactx2 += self.bpe_rnn2_Wcx(prior_att_c) # [n_bb,dunits]
                h2 = torch.tanh(preactx2)
                h2 = u2 * h1 + (1. - u2) * h2 
                # h2 = m_.unsqueeze(1) * h2 + (1. - m_).unsqueeze(1) * h1 # [n_bb,dunits]

                ## 0. RNN : s'(t)=RNN(y(t-1),s(t-1)) 
                # || to combine [emb(t-1),state(t-1)] -> predt-state(t) 
                # || RNN(dunits) : concat[ey,ctx-previous-posterior]
                preact3 = self.char2bpe_rnn1_U(hbpe_) # no bias -> [B,dunits*2]=[10,600]
                preact3 += c2b_x_ # [B,dunits*2]=[10,600]
                preact3 = torch.sigmoid(preact3)
                r3 = self._slice(preact3, 0, self.dunits) # reset gate [B,dunits]=[10,300]
                u3 = self._slice(preact3, 1, self.dunits) # update gate [B,dunits]=[10,300]
                preactx3 = self.char2bpe_rnn1_Ux(hbpe_) # no bias [B,dunits]=[10,300]
                preactx3 *= r3
                preactx3 += c2b_xx_ 
                h3 = torch.tanh(preactx3)
                h3 = u3 * hbpe_ + (1. - u3) * h3

                ## 2. Attention : c(t)=AttentionContext(s'(t), enc_o) 
                # || dunits (prior_att_c[B,eprojs] prior_att_w[B,T])
                c2b_prior_att_c, c2b_prior_att_w = self.att_char2bpe[att_bpe_idx](cc_, exp_hlens, h3, prev_c2b_alpha)

                ## 3. RNN : s(t)=RNN(c(t),s'(t)) 
                # || att_c(eprojs) z_infocomb_list(dunits)
                h3 = torch.sigmoid( self.c2b_statecomb(h1) ) * h3 + h1
                preact4 = self.char2bpe_rnn2_U_n1(h3) # [B,dunits*2]=[10,600]
                preact4 += self.char2bpe_rnn2_Wc(c2b_prior_att_c) # no bias [B,dunits*2]=[10,600]
                preact4 = torch.sigmoid(preact4)
                r4 = self._slice(preact4, 0, self.dunits) # [B,dunits]
                u4 = self._slice(preact4, 1, self.dunits) # [B,dunits]
                preactx4 = self.char2bpe_rnn2_Ux_n1(h3) # [B,dunits]=[10,300]
                preactx4 *= r4
                preactx4 += self.char2bpe_rnn2_Wcx(c2b_prior_att_c) # [B,dunits]=[10,300]
                h4 = torch.tanh(preactx4)
                h4 = u4 * h3 + (1. - u4) * h4

                ## Information combine for classification
                context = torch.cat([prior_att_c, c2b_prior_att_c],dim=1)
                cstates = torch.cat([h2, h4],dim=1)

                ## 4. Do classification for every frame
                logit_all_for_classif = self.classbpe_state_W(cstates) + self.classbpe_yprev_W(ey) + self.classbpe_enc_W(context) # [n_bb,dunits]
                similarb = logit_all_for_classif.size(0) # 
                logit_linear    = torch.reshape( logit_all_for_classif, (similarb, -1, 2) )
                logit_clf, _ = torch.max( logit_linear, dim=2 ) # [n_bb,dunits//2]
                probs = self.bpe_class_W(logit_clf) # [n_bb,odim]
                probs = F.softmax(probs, dim=1) # [n_bb,odim]

            # get nbest local scores and their ids
            local_scores = att_weight * torch.log(probs) # + log for score fusion [n_bb,odim]=[30,52]

            # rnnlm
            if rnnlm:
                rnnlm_state, local_lm_scores = rnnlm.buff_predict(rnnlm_prev, vy, n_bb)
                local_scores = local_scores + recog_args.lm_weight * local_lm_scores
            local_scores = local_scores.view(batch, n_bo) # [n_bb, odim] -> [batch*beam, odim] -> [batch, beam*odim]

            # ctc
            if lpz is not None:
                ctc_scores, ctc_states = ctc_prefix_score(yseq, ctc_states_prev, accum_odim_ids)
                ctc_scores = ctc_scores.view(batch, n_bo)
                local_scores = local_scores + ctc_weight * (ctc_scores - ctc_scores_prev)
            local_scores = local_scores.view(batch, beam, odim_nums) # [batch, beam, odim]

            if i == 0:
                local_scores[:, 1:, :] = self.logzero
            local_best_scores, local_best_odims = torch.topk(local_scores.view(batch, beam, odim_nums), beam, 2) # beam*senone -> beam*beam

            ###### char eos --> beam : ---> for prepare eos ######
            if (decode_choice == 'bpe'):
                if switch_addcode_forbi == False: 
                    local_scores_char = att_weight * self.char_onestep_decoder(exp_h, exp_hlens, subprev_alpha, hbpe_, yseq_char)
                else:
                    local_scores_char = att_weight * self.char_onestep_decoder(exp_h, exp_hlens, subprev_alpha, hbpe_, yseq_char, infor_bpe2char)                
                local_scores_char = local_scores_char.view(batch, beam, self.odim_char)
                # if i == 0: ############ ?????? ############
                #     local_scores_char[:, 1:, :] = self.logzero

            # local pruning (via xp)
            local_scores = np.full((n_bbo,), self.logzero)
            _best_odims = local_best_odims.view(n_bb, beam) + pad_o
            _best_odims = _best_odims.view(-1).cpu().numpy()
            _best_score = local_best_scores.view(-1).cpu().detach().numpy()
            local_scores[_best_odims] = _best_score
            local_scores = to_device(self, torch.from_numpy(local_scores).float()).view(batch, beam, odim_nums) # 这里应该是所有的可能得分[batch,beam,odim]

            ######################################################################################
            # calculate the eos score (maybe add the char score)
            if (decode_choice == 'bpe') and (CHOICE_SCORE_TYPES != "bpeonly"):
                eos_vscores      = local_scores[:, :, eos_token] + local_scores_char[:, :, self.eos_char] + vscores
                y_prev_char      = yseq_char[:][:]
            else:                
                eos_vscores      = local_scores[:, :, eos_token] + local_scores_char[:, :, self.eos_char]*0 + vscores # default setting (no extra char)
                y_prev_char      = yseq_char[:][:]
            ######################################################################################
            vscores = vscores.view(batch, beam, 1).repeat(1, 1, odim_nums) # [batch,beam,1] -> [batch,beam,52] -> 感觉像是开辟空间
            vscores[:, :, eos_token] = self.logzero
            vscores = (vscores + local_scores).view(batch, n_bo) # 这里是[1,beam*odim]

            # global pruning (pad_b is 0 when batch=1)
            accum_best_scores, accum_best_ids = torch.topk(vscores, beam, 1) # prune and choose 在整个beam*odim的范围中进行结果挑选
            accum_odim_ids = torch.fmod(accum_best_ids, odim_nums).view(-1).data.cpu().tolist() # 取余：查看每一个batch选的是哪一个输出字符 [wi]
            accum_padded_odim_ids = (torch.fmod(accum_best_ids, n_bo) + pad_bo).view(-1).data.cpu().tolist() # 筛选出哪一个token 这里实在整除            
            accum_padded_beam_ids  = (torch.div(accum_best_ids, odim_nums) + pad_b).view(-1).data.cpu().tolist() # 筛选出哪一个Beam
            accum_padded_odim_ids1 = (torch.fmod(accum_best_ids, odim_nums) + pad_b).view(-1).data.cpu().tolist() # 筛选出哪一个Beam(额外添加的，并不确定这里是否正确)

            y_prev = yseq[:][:]
            yseq = index_select_list(yseq, accum_padded_beam_ids) # 候选解码的所有序列
            yseq = append_ids(yseq, accum_odim_ids) # 添加挑选出来的label
            vscores = accum_best_scores # 活下来的beam的得分情况
            vidx = to_device(self, torch.LongTensor(accum_padded_beam_ids)) # Final idx[为什么只要保留batch的index]
            oidx = to_device(self, torch.LongTensor(accum_padded_odim_ids))

            ## 注意挑选1.postr_alpha需要两维挑选 
            # 2.prior_att_w需要进行一维的挑选 
            # 3.h2的挑选是进行一维的挑选 
            # 4.还有输出结果  
            # 5.还有对应的score情况  总共应该有五项结果 ##
            prev_alpha = torch.index_select(prior_att_w.view(n_bb, *prior_att_w.shape[1:]), 0, vidx) # 挑选n_bb(beam*batch)的结果 挑选中间结果(1): prior_alpha
            h_ = torch.index_select(h2.view(n_bb, *h2.shape[1:]), 0, vidx) # 挑选所要的中间结果(3): state h2(30,800)
            prev_c2b_alpha = torch.index_select(c2b_prior_att_w.view(n_bb, *c2b_prior_att_w.shape[1:]), 0, vidx) # 挑选n_bb(beam*batch)的结果 挑选中间结果(1): prior_alpha

            ## RNNLM Score
            if rnnlm:
                rnnlm_prev = index_select_lm_state(rnnlm_state, 0, vidx)
            ## CTC Score
            if lpz is not None:
                ctc_vidx = to_device(self, torch.LongTensor(accum_padded_odim_ids))
                ctc_scores_prev = torch.index_select(ctc_scores.view(-1), 0, ctc_vidx)
                ctc_scores_prev = ctc_scores_prev.view(-1, 1).repeat(1, odim_nums).view(batch, n_bo)

                ctc_states = torch.transpose(ctc_states, 1, 3).contiguous()
                ctc_states = ctc_states.view(n_bbo, 2, -1)
                ctc_states_prev = torch.index_select(ctc_states, 0, ctc_vidx).view(n_bb, 2, -1)
                ctc_states_prev = torch.transpose(ctc_states_prev, 1, 2)

            
            # 1. 首先yseq_char适用于存放所有的char历史的
            if (decode_choice == 'bpe'): 
                vscores_char = (torch.index_select(vscores_char.view(beam,-1), 0, vidx)).view(batch, beam)

                subprev_alpha = torch.index_select(subprev_alpha.view(n_bb, *subprev_alpha.shape[1:]), 0, vidx)
                hbpe_ = torch.index_select(hbpe_.view(n_bb, *hbpe_.shape[1:]), 0, vidx) ############ 新增裁剪过程

                yseq_char_inbpe = index_select_list(yseq_char_inbpe, accum_padded_beam_ids) # 重新调整 调整char对应的
                yseq_char       = index_select_list(yseq_char,       accum_padded_beam_ids)

                # bpe_token_list = [self.bpedict[idx] for idx in oidx.data.cpu().tolist()] # ['▁', '▁ECONOM', '▁E', '▁THE', 'A']
                bpe_token_list = [self.bpedict[idx] for idx in accum_odim_ids] # ['▁ALSO']
                # print('this is',bpe_token_list)

                max_len = 0
                char_len = np.zeros(len(bpe_token_list))
                for idx,bpe_token in enumerate(bpe_token_list):
                    if (i == 0) and bpe_token == "▁": ## 如果第一个字符是"▁", 那么：就选择忘记这个BPE的情况continue:
                        char_len[idx] = len(yseq_char_inbpe[idx]) - 1
                        max_len = max(max_len, char_len[idx])
                        continue                    
                    elif (i == 0) and (bpe_token[0] == "▁"):
                        token = bpe_token[1:]
                    else:
                        token = bpe_token
                    
                    if token == '[laughter]' or token == '[noise]' or token == '[vocalized-noise]' or token == '<eos>' or token == '<blank>' or token == '<unk>': # <unk> <blank> <eos>
                        yseq_char_inbpe[idx].append(self.inverse_chardict[token])
                        yseq_char[idx].append(self.inverse_chardict[token]) ## all length
                    else:
                        for iidx, str_ in enumerate(token):
                            yseq_char_inbpe[idx].append(self.inverse_chardict[str_])
                            yseq_char[idx].append(self.inverse_chardict[str_]) ## all length

                    char_len[idx] = len(yseq_char_inbpe[idx]) - 1
                    max_len = max(max_len, char_len[idx])

                ysequence = self.eos_char * np.ones((len(yseq_char_inbpe), int(max_len+1) ), dtype=np.int) # (10,5)            
                for idx,chartoken in enumerate(yseq_char_inbpe):
                    ysequence[idx,:len(chartoken)] = np.array(chartoken)
                ysequence = to_device(self, torch.from_numpy(ysequence))
                yseq_char_inbpe = [[chartoken[-1]] for chartoken in yseq_char_inbpe] # 在这里尝试保留为未下一个char解码时刻准备的 previous y.

                if switch_addcode_forbi == False: 
                    local_scores_char, subprev_alpha, hbpe_ = self.char_onebpe_decoder(exp_h, exp_hlens, subprev_alpha, hbpe_, ysequence, char_len) # yseq_char
                else:
                    local_scores_char, subprev_alpha, hbpe_ = self.char_onebpe_decoder(exp_h, exp_hlens, subprev_alpha, hbpe_, ysequence, char_len, infor_bpe2char) # yseq_char

            if CHOICE_SCORE_TYPES == "bpeonly":
                vscores += local_scores_char*0
            elif CHOICE_SCORE_TYPES == "bpe1char":
                vscores += local_scores_char
            elif CHOICE_SCORE_TYPES == "bpe2char":
                charmask = (char_len!=0.0).astype(np.float32)
                charmask = torch.from_numpy(( 1.0/(char_len+1e-10) )*charmask).to(local_scores_char)
                vscores += local_scores_char*charmask
            else:
                print("ERROR HERE FOR USING CHOICE_SCORE_TYPES %s" %(CHOICE_SCORE_TYPES))
                exit(0)

            # pick ended hyps
            # 1. 30
            # 2. thr[samp_i]:
            # 3. ended_hyps Useful eos sequence (*)
            if i > minlen:
                k = 0
                penalty_i = (i + 1) * penalty
                thr = accum_best_scores[:, -1] # 排序的最低一位的数值得分情况
                for samp_i in six.moves.range(batch): # 对于其中每一个batch进行修改
                    if stop_search[samp_i]:
                        k = k + beam
                        continue
                    for beam_j in six.moves.range(beam):
                        if eos_vscores[samp_i, beam_j] > thr[samp_i]:
                            yk = y_prev[k][:]
                            yk.append(eos_token)
                            if len(yk) < hlens[samp_i]:
                                _vscore = eos_vscores[samp_i][beam_j] + penalty_i
                                _score = _vscore.data.cpu().numpy()
                                normalize_score = False
                                if normalize_score:
                                    _vscore = _vscore / len(yk)
                                _score = _vscore.data.cpu().numpy()
                                ended_hyps[samp_i].append({'yseq': yk, 'vscore': _vscore, 'score': _score}) ## 
                        k = k + 1

            # end detection
            stop_search = [stop_search[samp_i] or end_detect(ended_hyps[samp_i], i)
                           for samp_i in six.moves.range(batch)]
            stop_search_summary = list(set(stop_search))
            if len(stop_search_summary) == 1 and stop_search_summary[0]:
                break

            torch.cuda.empty_cache()


        dummy_hyps = [{'yseq': [sos_token, eos_token], 'score': np.array([-float('inf')])}]
        
        ended_hyps = [ended_hyps[samp_i] if len(ended_hyps[samp_i]) != 0 else dummy_hyps
                      for samp_i in six.moves.range(batch)]
        nbest_hyps = [sorted(ended_hyps[samp_i], key=lambda x: x['score'], reverse=True)[:min(len(ended_hyps[samp_i]), recog_args.nbest)] # recog_args.nbest (1)
                      for samp_i in six.moves.range(batch)]
        print('[mltaskr] The value of CHOICE_SCORE_TYPES is %s. TJ-USING.' %(CHOICE_SCORE_TYPES))

        return nbest_hyps



    def char_onebpe_decoder(self, exp_h, exp_hlens, prev_alpha, h_, yseq, char_len, bpe2char_state=None, strm_idx=0):  # realmask查验
        # exp_h         ：torch.Tensor (beam,decT,dunit)
        # exp_hlens     ：list
        # previous_alpha：torch.Tensor (beam,decT)
        # previous_state：torch.Tensor (beam,dunit)
        # yseq          ：numpy.ndarray
        # char_len      ：numpy.ndarray
        att_char_idx = min(strm_idx, len(self.att_char) - 1)
        realmask = (to_device(self, make_trueframe_mask(char_len))).float()       # (1 is useful)
        n_bb = exp_h.shape[0]
        local_scores = torch.zeros((n_bb)).to(exp_h)

        for ii in range(int(max(char_len))):
            vy = yseq[:,ii] # to_device(self, torch.LongTensor((yseq[:,i])))         # get the last char
            ey = self.dropout_emb_char(self.embed_char(vy)) # [n_bb, embsize]
            xx_ = self.char_rnn1_Wx(ey) # state_belowx[n_bb, dunits]
            x_ = self.char_rnn1_W(ey) # [n_bb, 2*dunits]
            cc_ = exp_h # [n_bb, T, eprojs]
            m_ = realmask[:,ii] #

            ## 1. RNN eqn.(22): s'(t)=RNN(y(t-1),s(t-1)) 
            # || to combine [emb(t-1),state(t-1)] -> predt-state(t) 
            # || RNN(dunits) : concat[ey,ctx-previous-posterior]
            preact1 = self.char_rnn1_U(h_) # [n_bb,dunits*2]
            preact1 += x_ # [n_bb,dunits*2]
            preact1 = torch.sigmoid(preact1)
            r1 = self._slice(preact1, 0, self.dunits) # reset gate [n_bb,dunits]
            u1 = self._slice(preact1, 1, self.dunits) # update gate [n_bb,dunits]
            preactx1 = self.char_rnn1_Ux(h_) # [n_bb,dunits]
            preactx1 *= r1
            preactx1 += xx_ 
            h1 = torch.tanh(preactx1)
            h1 = u1 * h_ + (1. - u1) * h1
            h1 = m_.unsqueeze(1) * h1 + (1. - m_).unsqueeze(1) * h_

            ## 2. Attention eqn.(23): c(t)=AttentionContext(s'(t), enc_o) 
            # || context vectot / attention weight (dunits)
            prior_att_c, prior_att_w = self.att_char[att_char_idx](cc_, exp_hlens, h1, prev_alpha)

            ## 3. RNN eqn.(24): s(t)=RNN(c(t),s'(t)) 
            # || att_c(eprojs) z_infocomb_list(dunits)
            preact2 = self.char_rnn2_U_n1(h1) # [n_bb,dunits*2]
            preact2 += self.char_rnn2_Wc(prior_att_c) # [n_bb,dunits*2]
            preact2 = torch.sigmoid(preact2)
            r2 = self._slice(preact2, 0, self.dunits) # [n_bb,dunits]
            u2 = self._slice(preact2, 1, self.dunits) # [n_bb,dunits]            
            preactx2 = self.char_rnn2_Ux_n1(h1) # [n_bb,dunits]
            preactx2 *= r2
            preactx2 += self.char_rnn2_Wcx(prior_att_c) # [n_bb,dunits]
            h2 = torch.tanh(preactx2)
            h2 = u2 * h1 + (1. - u2) * h2 
            h2 = m_.unsqueeze(1) * h2 + (1. - m_).unsqueeze(1) * h1 # [n_bb,dunits]

            ## 4. Do classification for every frame
            if switch_addcode_forbi == False: 
                logit_all_for_classif = self.classchar_state_W(h2) + self.classchar_yprev_W(ey) + self.classchar_enc_W(prior_att_c) # [n_bb,dunits]
            else:
                shortline = self.classchar_state_W(h2) + self.classchar_yprev_W(ey) + self.classchar_enc_W(prior_att_c)
                branch_pass = torch.relu( ( self.classb2c_state_W1(shortline) ) )
                branch_pass = ( self.classb2c_state_W2(branch_pass) )
                logit_all_for_classif = shortline + branch_pass # [nbatch,dunits]          
            ## ## ## ##
            similarb = logit_all_for_classif.size(0)                                                                            # 
            logit_linear    = torch.reshape( logit_all_for_classif, (similarb, self.dunits//2, 2) )
            logit_clf, _ = torch.max( logit_linear, dim=2 )                                                                     # [n_bb,dunits//2]
            probs = self.char_class_W(logit_clf)                                                                                # [n_bb,odim]
            probs = F.softmax(probs, dim=1)                                                                                     # [n_bb,odim]

            ## get output result
            prev_alpha = m_.unsqueeze(1) * prior_att_w + (1. - m_).unsqueeze(1) * prev_alpha                                    # [n_bb,dunits]
            true_choose_mask = slow_onehot(labels=yseq[:,ii+1], senone=self.odim_char)                                          # -1 means removed

            local_scores += (torch.log(probs) * true_choose_mask).sum(1) * m_                                                   # [batch*beam,odim]->[batch*beam] 处理长度问题再和外界处理
            h_ = h2

        return local_scores, prev_alpha, h_



    def char_onestep_decoder(self, exp_h, exp_hlens, prev_alpha, h_, yseq, bpe2char_state=None, strm_idx=0):
        att_char_idx = min(strm_idx, len(self.att_char) - 1)

        vy = to_device(self, torch.LongTensor(get_last_yseq(yseq))) # get the last char
        ey = self.dropout_emb_char(self.embed_char(vy)) # [n_bb, embsize]
        xx_ = self.char_rnn1_Wx(ey) # state_belowx[n_bb, dunits]
        x_ = self.char_rnn1_W(ey) # [n_bb, 2*dunits]
        cc_ = exp_h # [n_bb, T, eprojs]

        ## 1. RNN eqn.(22): s'(t)=RNN(y(t-1),s(t-1)) || to combine [emb(t-1),state(t-1)] -> predt-state(t) || RNN(dunits) : concat[ey,ctx-previous-posterior]
        preact1 = self.char_rnn1_U(h_) # [n_bb,dunits*2]
        preact1 += x_ # [n_bb,dunits*2]
        preact1 = torch.sigmoid(preact1)
        r1 = self._slice(preact1, 0, self.dunits) # reset gate [n_bb,dunits]
        u1 = self._slice(preact1, 1, self.dunits) # update gate [n_bb,dunits]
        preactx1 = self.char_rnn1_Ux(h_) # [n_bb,dunits]
        preactx1 *= r1
        preactx1 += xx_ 
        h1 = torch.tanh(preactx1)
        h1 = u1 * h_ + (1. - u1) * h1
        # h1 = m_.unsqueeze(1) * h1 + (1. - m_).unsqueeze(1) * h_

        ## 2. Attention eqn.(23): c(t)=AttentionContext(s'(t), enc_o) || dunits
        prior_att_c, prior_att_w = self.att_char[att_char_idx](cc_, exp_hlens, h1, prev_alpha)

        ## 3. RNN eqn.(24): s(t)=RNN(c(t),s'(t)) || att_c(eprojs) z_infocomb_list(dunits)
        preact2 = self.char_rnn2_U_n1(h1) # [n_bb,dunits*2]
        preact2 += self.char_rnn2_Wc(prior_att_c) # [n_bb,dunits*2]
        preact2 = torch.sigmoid(preact2)
        r2 = self._slice(preact2, 0, self.dunits) # [n_bb,dunits]
        u2 = self._slice(preact2, 1, self.dunits) # [n_bb,dunits]            
        preactx2 = self.char_rnn2_Ux_n1(h1) # [n_bb,dunits]
        preactx2 *= r2
        preactx2 += self.char_rnn2_Wcx(prior_att_c) # [n_bb,dunits]
        h2 = torch.tanh(preactx2)
        h2 = u2 * h1 + (1. - u2) * h2 
        # h2 = m_.unsqueeze(1) * h2 + (1. - m_).unsqueeze(1) * h1 # [n_bb,dunits]

        ## 4. Do classification for every frame
        if switch_addcode_forbi == False: 
            logit_all_for_classif = self.classchar_state_W(h2) + self.classchar_yprev_W(ey) + self.classchar_enc_W(prior_att_c) # [n_bb,dunits]
        else:
            shortline = self.classchar_state_W(h2) + self.classchar_yprev_W(ey) + self.classchar_enc_W(prior_att_c)
            branch_pass = torch.relu( ( self.classb2c_state_W1(shortline) ) )
            branch_pass = ( self.classb2c_state_W2(branch_pass) )
            logit_all_for_classif = shortline + branch_pass # [nbatch,dunits]
        ## ## ##
        similarb = logit_all_for_classif.size(0) # 
        logit_linear    = torch.reshape( logit_all_for_classif, (similarb, self.dunits//2, 2) )
        logit_clf, _ = torch.max( logit_linear, dim=2 ) # [n_bb,dunits//2]
        probs = self.char_class_W(logit_clf) # [n_bb,odim]
        probs = F.softmax(probs, dim=1) # [n_bb,odim]

        return torch.log(probs)



# this commands function
def decoder_base_mltaskllers_for(
                    args, odim_char, odim_bpe, 
                    sos_char, eos_char, 
                    sos_bpe, eos_bpe, 
                    att_char, att_bpe, att_char2bpe,
                    labeldist_char, labeldist_bpe):
    return MLSoftmaxDecoder(
                args.eprojs, args.dtype, args.dlayers, args.dunits, 
                odim_char, odim_bpe, 
                sos_char, eos_char, sos_bpe, eos_bpe, 
                att_char, att_bpe, att_char2bpe, args, 
                args.verbose, args.char_list, args.bpe_list, 
                labeldist_char, labeldist_bpe, 
                args.lsm_weight, args.sampling_probability, args.dropout_rate_decoder)
