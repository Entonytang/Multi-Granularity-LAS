#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# this code is for end-2end speech recognition;(based on espnet)
#
# Authors: Jian Tang (modification)
'''
Build a encoder&decdoer model with soft attention
    (3) 这里是要实现nmt_softmax的convF-attention的做法
    这个程序是为了反向验证Posterior在代码上是否有问题：在训练-解码的过程中
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
from espnet.nets.pytorch_backend.nets_utils import make_truemask
MAX_DECODER_OUTPUT = 5 # ?
CTC_SCORING_RATIO = 1.5 # ?
traditional_loss_calculate = False
xinput_operation = True

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


def kullback_leibler_divergence_masked(y_true, y_pred, masked, axis=-1):
    y_true = torch.clamp(y_true, 1e-8, 1)
    y_pred = torch.clamp(y_pred, 1e-8, 1)
    return torch.sum(masked * y_true * torch.log(y_true / y_pred), dim=axis)    


class IntergrationPostrPriorDecoder(torch.nn.Module):
    """Decoder module 
        :param int eprojs: # encoder projection units [D_enc]
        :param int odim: dimension of outputs 
        :param str dtype: gru or lstm
        :param int dlayers: # decoder layers [decoder RNN]
        :param int dunits: # decoder units
        :param int sos: start of sequence symbol id
        :param int eos: end of sequence symbol id
        :param torch.nn.Module att: attention module [attention]
        :param int verbose: verbose level
        :param list char_list: list of character strings
        :param ndarray labeldist: distribution of label smoothing
        :param float lsm_weight: label smoothing weight
        :param float sampling_probability: scheduled sampling probability [schedule sampling]
        :param float dropout: dropout rate [dropout]
    """

    def __init__(self, eprojs, odim, dtype, dlayers, dunits, sos, eos, att, 
                    verbose=0, char_list=None, labeldist=None, lsm_weight=0., sampling_probability=0.0, dropout=0.0, embsize=256, 
                    dec_filts=1, dist_clip=0.1):
        super(IntergrationPostrPriorDecoder, self).__init__()
        self.dtype = dtype
        self.dunits = dunits
        self.dlayers = dlayers
        self.embsize = embsize
        self.dist_clip = dist_clip
        self.embed = torch.nn.Embedding(odim, self.embsize)
        self.dropout_emb = torch.nn.Dropout(p=dropout)
        
        ## Split RNN Information Combination Layer ##
        self.rnn1_Wx = torch.nn.Linear(embsize, dunits)
        self.rnn1_W = torch.nn.Linear(embsize, dunits*2)
        self.rnn1_U = torch.nn.Linear(dunits, dunits*2, bias=False)
        self.rnn1_Ux = torch.nn.Linear(dunits, dunits, bias=False)
        self.rnn1_context_Wx = torch.nn.Linear(eprojs, dunits)
        self.rnn1_context_W = torch.nn.Linear(eprojs, dunits*2)
        self.loc_conv = torch.nn.Conv2d(self.dunits, odim, (1, 2 * dec_filts + 1), padding=(0, dec_filts), bias=False)
                
        ## Split RNN for next iterator layer ##
        self.rnn2_U_n1 = torch.nn.Linear(dunits, dunits*2)
        self.rnn2_Wc = torch.nn.Linear(eprojs, dunits*2, bias=False)
        self.rnn2_Ux_n1 = torch.nn.Linear(dunits, dunits)
        self.rnn2_Wcx = torch.nn.Linear(eprojs, dunits, bias=False)
        ## New added here by jtang ##
        self.init_state_fc = torch.nn.Linear(eprojs, dunits)
        ## Classification Part ##
        self.classif_state_W = torch.nn.Linear(dunits, odim)
        self.classif_y_prev_W = torch.nn.Linear(self.embsize, odim, bias=False)
        self.classif_encoder_W = torch.nn.Linear(eprojs, odim, bias=False)
        print('Setting loss calc', traditional_loss_calculate)
        print('Decoder version --- decoders posterior[postrv2cmt postr_alpha_true]. + [Local Convolution].')
        #########################
        self.ignore_id = -1

        self.loss = None
        self.att = att # what (maybe the attention layer)
        self.dunits = dunits
        self.sos = sos
        self.eos = eos
        self.odim = odim
        self.verbose = verbose
        self.char_list = char_list
        # for label smoothing 
        self.labeldist = labeldist # ?
        self.vlabeldist = None # ?
        self.lsm_weight = lsm_weight 
        self.sampling_probability = sampling_probability # focus 0 means all true-label 
        self.dropout = dropout

        self.logzero = -10000000000.0

    # initial decoder state  enc_o[B,T,D] -> mean -> (B,D_enc) -> FC -> [B,D_dec]
    def mean_init_state(self, hs_pad, hlens):
        hlens = (torch.Tensor(hlens)).float().to(hs_pad)                             # training process
        mean_enc_o = torch.sum(input=hs_pad, dim=1)  # [B,D_enc]
        result = torch.div( mean_enc_o, hlens.view(hs_pad.size(0), 1) ) # [B,1]
        result = self.init_state_fc(result)
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


    def forward(self, hs_pad, hlens, ys_pad, penalty_weight=0, strm_idx=0): 
        """Decoder forward

            :param torch.Tensor hs_pad: batch of padded hidden state sequences (B, Tmax, D) : enc_o
            :param torch.Tensor hlens: batch of lengths of hidden state sequences (B) : enc_o_len
            :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax) : true_label
            :param int strm_idx: stream index indicates the index of decoding stream. (y_idx ? )
            :return: attention loss value
            :rtype: torch.Tensor
            :return: accuracy
            :rtype: float
        """
        # TODO(kan-bayashi): need to make more smart way
        ys = [y[y != self.ignore_id] for y in ys_pad]  # parse padded ys
        # attention index for the attention module in SPA (speaker parallel attention), att_idx is used to select attention module. In other cases, it is 0.
        att_idx = min(strm_idx, len(self.att) - 1) # useful only for SAT

        # hlen should be list of integer
        hlens = list(map(int, hlens))

        self.loss = None
        # prepare input and output word sequences with sos/eos IDs
        eos = ys[0].new([self.eos])
        sos = ys[0].new([self.sos])
        ys_in = [torch.cat([sos, y], dim=0) for y in ys] # in this way : the label and the feedback y is created (ys_in is for state(t-1))
        ys_out = [torch.cat([y, eos], dim=0) for y in ys] #  

        ## 先生成deocder_len的序列
        ys_out_len = [y.size(0) for y in ys_out]
        ymask = (to_device(self, make_trueframe_mask(ys_out_len))).float()
        context_mask = (to_device(self, make_trueframe_mask(hlens))).float() # [B,T]=[10,203]

        # padding for ys with -1
        # pys: utt x olen
        ys_in_pad = pad_list(ys_in, self.eos) # pad feature
        ys_out_pad = pad_list(ys_out, self.ignore_id) # pad output symbol
        ys_true_pad = pad_list(ys_out, self.eos) # pad output symbol

        # get dim, length info [B,T_dec] = [15,168]
        batch = ys_out_pad.size(0)
        olength = ys_out_pad.size(1)
        logging.info(self.__class__.__name__ + ' input lengths:  ' + str(hlens))
        logging.info(self.__class__.__name__ + ' output lengths: ' + str([y.size(0) for y in ys_out])) # verbose 0

        # initialization : give initial context_vector & decoder_state
        h_ = self.mean_init_state(hs_pad, hlens) # self.zero_state(hs_pad)
        # prev_alpha = None # previous prior attention score
        prev_postr_energy = (to_device(self, make_trueframe_mask(hlens))).float() # previous postr attention score
        prev_alpha = prev_postr_energy / (prev_postr_energy.sum(1, keepdim=True)+1e-8) # [B,T]
        next_probs = (1.0/self.odim)*torch.ones((batch, self.odim)).to(prev_alpha)

        self.att[att_idx].reset()  # reset pre-computation of h

        # pre-computation of embedding y -> emb(y) [B,C,D_senone]
        eys = self.dropout_emb(self.embed(ys_in_pad))  # utt x olen x zdim [B,decoder_T,300]

        # Posterior Attention & Decoder Process
        y_predict_all = []
        distance_all = []
        
        # emb : [B, decoder_T, embsize]
        state_below = eys                # emb -> y(t-1): [B,decoder_T,embsize]=[B,144,256]
        state_below_ = self.rnn1_W(eys)  # [B,decoder_T,dunits]=[10,144,600]
        state_belowx = self.rnn1_Wx(eys) # [B,decoder_T,dunits]=[10,144,300]
        if xinput_operation:
            enc_out_transform = self.classif_encoder_W(hs_pad) # [B,T,D]
            hs_pad_input = hs_pad.transpose(2,1).unsqueeze(2) # [B,T,D]->[B,D,T]->[B,D,1,T] 
            x_conv = self.loc_conv( hs_pad_input ).squeeze(2).transpose(2,1) # [B,D,1,T]->[B,D,T]->[B,T,D]
            enc_out_transform = enc_out_transform * torch.sigmoid(x_conv)
        else:
            enc_out_transform = self.classif_encoder_W(hs_pad)

        ilength = hs_pad.size(1)
        for i in six.moves.range(olength):  # For every characters
            m_ = ymask[:,i] # MASK the variables for this char decoder steps
            x_ = state_below_[:,i,:]
            xx_ = state_belowx[:,i,:]
            cc_ = hs_pad

            ## 1. RNN eqn.(22): s'(t)=RNN(y(t-1),s(t-1)) || to combine [emb(t-1),state(t-1)] -> predt-state(t) || RNN(dunits) : concat[ey,ctx-previous-posterior]
            prev_context_vector = torch.sum(prev_alpha.unsqueeze(2)*cc_, dim=1)
            preact1 = self.rnn1_U(h_) # no bias -> [B,dunits*2]=[10,600]
            preact1 += x_ # [B,dunits*2]=[10,600]
            preact1 += self.rnn1_context_W( prev_context_vector ) # [B,T,eprojs][eprojs,dunits*2]
            preact1 = torch.sigmoid(preact1)
            r1 = self._slice(preact1, 0, self.dunits) # reset gate [B,dunits]=[10,300]
            u1 = self._slice(preact1, 1, self.dunits) # update gate [B,dunits]=[10,300]
            preactx1 = self.rnn1_Ux(h_) # no bias [B,dunits]=[10,300]
            preactx1 *= r1
            preactx1 += xx_ 
            preactx1 += self.rnn1_context_Wx( prev_context_vector )
            h1 = torch.tanh(preactx1)
            h1 = u1 * h_ + (1. - u1) * h1
            h1 = m_.unsqueeze(1) * h1 + (1. - m_).unsqueeze(1) * h_

            ## 2. Attention eqn.(23): c(t)=AttentionContext(s'(t), enc_o) || dunits (prior_att_c[B,eprojs] prior_att_w[B,T])
            prior_att_c, prior_att_w = self.att[att_idx](cc_, hlens, h1, prev_alpha) 
            
            ## 3. RNN eqn.(24): s(t)=RNN(c(t),s'(t)) || att_c(eprojs) z_infocomb_list(dunits)
            preact2 = self.rnn2_U_n1(h1) # [B,dunits*2]=[10,600]
            preact2 += self.rnn2_Wc(prior_att_c) # no bias [B,dunits*2]=[10,600]
            preact2 = torch.sigmoid(preact2)
            r2 = self._slice(preact2, 0, self.dunits) # [B,dunits]
            u2 = self._slice(preact2, 1, self.dunits) # [B,dunits]
            preactx2 = self.rnn2_Ux_n1(h1) # [B,dunits]=[10,300]
            preactx2 *= r2
            preactx2 += self.rnn2_Wcx(prior_att_c) # [B,dunits]=[10,300]
            h2 = torch.tanh(preactx2)
            h2 = u2 * h1 + (1. - u2) * h2
            h2 = m_.unsqueeze(1) * h2 + (1. - m_).unsqueeze(1) * h1 # [B,dunits]=[10,300]

            ## 4. Classification for every frames
            logit = self.classif_state_W(h2) + self.classif_y_prev_W(state_below[:, i, :]) # [B,dunits]=[10,300]
            logit_all_for_classif = logit.unsqueeze(1) + enc_out_transform # [B,T,dunits]=[10,203,300]
            
            nframes = logit_all_for_classif.size(1) # T 203
            probs = logit_all_for_classif # [B,T,52]

            probs = torch.reshape(probs, (batch*nframes, self.odim))
            probs = F.softmax(probs, dim=-1) 
            probs = torch.reshape(probs, (batch, nframes, self.odim)) # [B,T,odim]=[10,203,52]
            next_probs = torch.sum(probs * (prior_att_w.unsqueeze(2) + 1e-8), dim=1) # [B,T,odim]->[B,52]

            ## 5.create postr attention scores
            minitiny = np.finfo('float32').tiny
            choose_topK = 1
            next_probs_argsort = torch.argsort(next_probs, dim=1, descending=False) # default is 0 -> 1
            topK_index = (next_probs_argsort[:,-1*choose_topK:]).long() # [B,odim]->[B]
            for_index_add = torch.arange(batch).long().to(topK_index)*self.odim
            topK_index = topK_index + for_index_add[:,None] # [B,topK]=[B]
            saved_index = topK_index.reshape([batch, choose_topK]) # [B,topK]
            padding = ((saved_index[:,0]).unsqueeze(1).repeat(1, self.odim-choose_topK)).long() # [B,odim-topK]
            all_index = torch.cat( [saved_index, padding], dim=1 ).long() # [B,odim]
            all_index = all_index.reshape([batch*self.odim])
            get_mask = torch.zeros((batch*self.odim)).to(next_probs)
            get_mask = get_mask.index_fill_(0, all_index, 1)
            get_mask = torch.reshape(get_mask, (batch, self.odim)) # [B,self.odim]
            masked_probs_energy = torch.exp( \
                torch.log(torch.clamp(probs * get_mask.unsqueeze(1), minitiny, 1.)) + \
                torch.log(torch.clamp(prior_att_w.unsqueeze(2),      minitiny, 1.)) \
                ) # [B,T,odim]*[B,None,odim]*[B,T,None]
            masked_probs_energy = torch.clamp(masked_probs_energy, minitiny, 1.)
            masked_probs_energy = torch.sum(masked_probs_energy, dim=2) # [B,T]
            masked_probs_energy = masked_probs_energy*context_mask
            postr_alpha = masked_probs_energy / (minitiny + masked_probs_energy.sum(1, keepdim=True)) # True posterior attention part (B,T) 

            ## 6.create postr attention scores
            true_choose_mask = slow_onehot(labels=ys_true_pad[:,i], senone=self.odim)
            masked_probs_energy = torch.exp( \
                torch.log(torch.clamp(probs * true_choose_mask.unsqueeze(1), minitiny, 1.)) + \
                torch.log(torch.clamp(prior_att_w.unsqueeze(2), minitiny, 1.)) \
                ) # [B,T,odim]
            masked_probs_energy = torch.clamp(masked_probs_energy, minitiny, 1.)
            masked_probs_energy = torch.sum(masked_probs_energy, dim=2) # [B,T]
            masked_probs_energy = masked_probs_energy*context_mask
            postr_alpha_true = masked_probs_energy / (minitiny + masked_probs_energy.sum(1, keepdim=True)) # True posterior attention part (B,T)    

            ## 7. pred posterior attention score
            distance = kullback_leibler_divergence(postr_alpha_true, postr_alpha, axis=1) # (B,T)
            distance = torch.where( distance <= self.dist_clip, torch.zeros(distance.size()).to(distance), distance) 
            distance_all.append(distance)

            ## 8. update previous variables and collect all y_predict/postr_ytrue/postr_pred
            y_predict_all.append( next_probs ) # [B,odim] ->
            prev_alpha = postr_alpha_true
            h_ = h2

        # ## 8. Get mainly loss : CE and auxiliary loss : KL-penalty
        # postr_pred_att_weights = torch.stack(postr_pred_att_weights, dim=1)
        # postr_true_att_weights = torch.stack(postr_true_att_weights, dim=1)
        # h2_all = torch.stack(h2_all, dim=1) # [B,decoder_T, dunits]

        if traditional_loss_calculate == True:
            dist_loss = torch.mean(distance_all)
            y_predict_all   = torch.stack(y_predict_all, dim=1).view(batch * olength, self.odim)   # [B,decoder_T, odim]->[B*decoder_T, odim]
            reduction_str = 'elementwise_mean' if LooseVersion(torch.__version__) < LooseVersion('1.0') else 'mean'
            # input : [N,C] label : [N] softmax is in F.cross_entropy (o:scalar)
            self.loss = F.categorical_crossentropy(y_predict_all, ys_out_pad.view(-1), ignore_index=self.ignore_id, reduction=reduction_str)
            self.loss += penalty_weight*dist_loss
            # -1: eos, which is removed in the loss computation
            self.loss *= (np.mean([len(x) for x in ys_in]) - 1)
            # pad_outputs: prediction tensors (B*decoder_T, D)
            # pad_targets: target tensors (B, decoder_T, D)
            acc = th_accuracy(y_predict_all, ys_out_pad, ignore_label=self.ignore_id)
            penalty =  dist_loss * (np.mean([len(x) for x in ys_in]) - 1)
            logging.info( 'With penalty '+str(penalty_weight)+'att loss:' + ''.join(str(self.loss.item()).split('\n')) )
        else:
            distance_all = torch.stack(distance_all, dim=1) * ymask # [B]->[B,decoder_T]            
            y_predict_all = torch.stack(y_predict_all, dim=1).view(batch * olength, self.odim)
            self.loss = F.categorical_crossentropy(y_predict_all, ys_out_pad.contiguous().view(-1).long(), ignore_index=self.ignore_id, reduction='none')
            self.loss = self.loss.view(batch, olength) 
            self.loss = self.loss + penalty_weight * distance_all
            self.loss = self.loss.sum(1)
            self.loss = self.loss.mean(0) # to be scalar 对比过theano的处理：和theano已经大体一致 torch:33.8797/theano=33.88
            penalty = (distance_all.sum(1)).mean(0)
            acc = th_accuracy(y_predict_all, ys_out_pad, ignore_label=self.ignore_id)
            logging.info( 'With penalty '+str(penalty_weight)+'att loss:' + ''.join(str(self.loss.item()).split('\n')) )

        # show predicted character sequence for debug (not changed it)
        if self.verbose > 0 and self.char_list is not None:
            ys_hat = y_predict_all.view(batch, olength, -1)
            ys_true = ys_out_pad
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

        if self.labeldist is not None:
            if self.vlabeldist is None:
                self.vlabeldist = to_device(self, torch.from_numpy(self.labeldist))

            loss_reg = - torch.sum((torch.log(y_predict_all) * self.vlabeldist).view(-1), dim=0) / len(ys_in)
            self.loss = (1. - self.lsm_weight) * self.loss + self.lsm_weight * loss_reg

        return self.loss, acc, penalty

    
    def recognize_beam(self, h, lpz, recog_args, char_list, rnnlm=None, strm_idx=0): # h(enc_o) [T,D_enc] lpz(CTC output)
        # remove sos
        return 0

    
    def recognize_beam_batch(self, h, hlens, lpz, recog_args, char_list, rnnlm=None, normalize_score=True, strm_idx=0):
        logging.info('input lengths: ' + str(h.size(1)))                                                        # [B,T,feadim]
        att_idx = min(strm_idx, len(self.att) - 1)                                                              # speaker adapation choice
        h = mask_by_length(h, hlens, 0.0)                                                                       # mask feature

        # search params
        batch = len(hlens)                                                                                      # 
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = recog_args.ctc_weight
        att_weight = 1.0 - ctc_weight

        n_bb = batch * beam                                                                                     # the true batch input (the beamsearch intermediate size)
        n_bo = beam * self.odim                                                                                 # the true batch senone
        n_bbo = n_bb * self.odim
        pad_b = to_device(self, torch.LongTensor([i * beam for i in six.moves.range(batch)]).view(-1, 1))       # stable
        pad_bo = to_device(self, torch.LongTensor([i * n_bo for i in six.moves.range(batch)]).view(-1, 1))      # 
        pad_o = to_device(self, torch.LongTensor([i * self.odim for i in six.moves.range(n_bb)]).view(-1, 1))   # 

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

        rnnlm_prev = None
        self.att[att_idx].reset()  # reset pre-computation of h

        yseq = [[self.sos] for _ in six.moves.range(n_bb)]
        accum_odim_ids = [self.sos for _ in six.moves.range(n_bb)]
        stop_search = [False for _ in six.moves.range(batch)]
        nbest_hyps = [[] for _ in six.moves.range(batch)]
        ended_hyps = [[] for _ in range(batch)]

        exp_hlens = hlens.repeat(beam).view(beam, batch).transpose(0, 1).contiguous()        
        exp_hlens = exp_hlens.view(-1).tolist()
        exp_h = h.unsqueeze(1).repeat(1, beam, 1, 1).contiguous()
        exp_h = exp_h.view(n_bb, h.size()[1], h.size()[2]) # [n_bb, T, eprojs]

        # extra initialization        
        h_ = self.mean_init_state(exp_h, list(map(int, exp_hlens)))#.repeat(n_bb, 1) # [n_bb, dunits]
        prev_postr_energy = (to_device(self, make_trueframe_mask(exp_hlens))).float() # previous postr attention score
        prev_alpha = ( prev_postr_energy / (prev_postr_energy.sum(1, keepdim=True)+1e-8) ) # [n_bb, ]
        next_probs = (1.0/self.odim)*torch.ones((n_bb, self.odim)).to(prev_alpha)        
        context_mask = (to_device(self, make_trueframe_mask(exp_hlens))).float() # [n_bb,T]
        # mask 采用全1的初始化方式,因为batch=1恒成立,所以不必担心这一点

        if lpz is not None: # no need to modification
            device_id = torch.cuda.device_of(next(self.parameters()).data).idx
            ctc_prefix_score = CTCPrefixScoreTH(lpz, 0, self.eos, beam, exp_hlens, device_id)
            ctc_states_prev = ctc_prefix_score.initial_state()
            ctc_scores_prev = to_device(self, torch.zeros(batch, n_bo))

        prior_att_context = []
        prior_att_weights = []
        if xinput_operation:
            enc_out_transform = self.classif_encoder_W(exp_h) # [B,T,D]
            hs_pad_input = exp_h.transpose(2,1).unsqueeze(2) # [B,T,D]->[B,D,T]->[B,D,1,T] 
            x_conv = self.loc_conv( hs_pad_input ).squeeze(2).transpose(2,1) # [B,D,1,T]->[B,D,T]->[B,T,D]
            enc_out_transform = enc_out_transform * torch.sigmoid(x_conv)            
        else:
            enc_out_transform = self.classif_encoder_W(exp_h)

        for i in six.moves.range(maxlen):
            logging.debug('position ' + str(i))
            vy = to_device(self, torch.LongTensor(get_last_yseq(yseq))) # get the last char
            ey = self.dropout_emb(self.embed(vy)) # [n_bb, embsize]
            xx_ = self.rnn1_Wx(ey) # state_belowx[n_bb, dunits]
            x_ = self.rnn1_W(ey) # [n_bb, 2*dunits]
            cc_ = exp_h # [n_bb, T, eprojs]

            ## 1. RNN eqn.(22): s'(t)=RNN(y(t-1),s(t-1)) || to combine [emb(t-1),state(t-1)] -> predt-state(t) || RNN(dunits) : concat[ey,ctx-previous-posterior]
            prev_context_vector = torch.sum(prev_alpha.unsqueeze(2)*exp_h, dim=1)
            preact1 = self.rnn1_U(h_) # [n_bb,dunits*2]
            preact1 += x_ # [n_bb,dunits*2]
            preact1 += self.rnn1_context_W( prev_context_vector ) # [B,T,eprojs][eprojs,dunits*2]
            preact1 = torch.sigmoid(preact1)
            r1 = self._slice(preact1, 0, self.dunits) # reset gate [n_bb,dunits]
            u1 = self._slice(preact1, 1, self.dunits) # update gate [n_bb,dunits]
            preactx1 = self.rnn1_Ux(h_) # [n_bb,dunits]
            preactx1 *= r1
            preactx1 += xx_ 
            preactx1 += self.rnn1_context_Wx( prev_context_vector ) 
            h1 = torch.tanh(preactx1)
            h1 = u1 * h_ + (1. - u1) * h1
            # h1 = m_.unsqueeze(1) * h1 + (1. - m_).unsqueeze(1) * h_

            ## 2. Attention eqn.(23): c(t)=AttentionContext(s'(t), enc_o) || dunits
            prior_att_c, prior_att_w = self.att[att_idx](cc_, exp_hlens, h1, prev_alpha) 
            prior_att_weights.append(prior_att_w)
            prior_att_context.append(prior_att_c)   

            ## 3. RNN eqn.(24): s(t)=RNN(c(t),s'(t)) || att_c(eprojs) z_infocomb_list(dunits)
            preact2 = self.rnn2_U_n1(h1) # [n_bb,dunits*2]
            preact2 += self.rnn2_Wc(prior_att_c) # [n_bb,dunits*2]
            preact2 = torch.sigmoid(preact2)
            r2 = self._slice(preact2, 0, self.dunits) # [n_bb,dunits]
            u2 = self._slice(preact2, 1, self.dunits) # [n_bb,dunits]            
            preactx2 = self.rnn2_Ux_n1(h1) # [n_bb,dunits]
            preactx2 *= r2
            preactx2 += self.rnn2_Wcx(prior_att_c) # [n_bb,dunits]
            h2 = torch.tanh(preactx2)
            h2 = u2 * h1 + (1. - u2) * h2 
            # h2 = m_.unsqueeze(1) * h2 + (1. - m_).unsqueeze(1) * h1 # [n_bb,dunits]

            ## 4. Do classification for every frame
            logit = self.classif_state_W(h2) + self.classif_y_prev_W(ey) # [B,dunits]=[10,300]
            logit_all_for_classif = logit.unsqueeze(1) + enc_out_transform # [B,T,dunits]=[10,203,300]
            
            similarb = logit_all_for_classif.size(0) # 
            nframes = logit_all_for_classif.size(1) # T 203
            probs = logit_all_for_classif

            probs = torch.reshape(probs, (similarb*nframes, self.odim))
            probs = F.softmax(probs, dim=-1) 
            probs = torch.reshape(probs, (similarb, nframes, self.odim)) # [B,T,odim]=[10,203,52]
            next_probs = torch.sum(probs * (prior_att_w.unsqueeze(2) + 1e-8), dim=1) # [B,T,odim]->[B,52]

            ## 5.create postr attention scores(修正：尝试老的实现方式)
            minitiny = np.finfo('float32').tiny
            masked_probs_energy = torch.exp( \
                torch.log(torch.clamp(probs, minitiny, 1.)) + \
                torch.log(torch.clamp(prior_att_w.unsqueeze(2), minitiny, 1.)) \
                )  # [n_bb,T,odim]
            masked_probs_energy = torch.clamp(masked_probs_energy, minitiny, 1.)
            masked_probs_energy = masked_probs_energy*context_mask.unsqueeze(2) # [n_bb, T, odim]            
            postr_att_w = masked_probs_energy / (minitiny + masked_probs_energy.sum(1, keepdim=True)) # Predict posterior attention part [n_bb, T, odim]

            # get nbest local scores and their ids                                                                                                              # 
            local_scores = att_weight * torch.log(next_probs) # + log for score fusion [n_bb,odim]=[30,52]

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
            local_scores = local_scores.view(batch, beam, self.odim) # [batch, beam, odim]

            if i == 0:
                local_scores[:, 1:, :] = self.logzero
            local_best_scores, local_best_odims = torch.topk(local_scores.view(batch, beam, self.odim), beam, 2)

            # local pruning (via xp)
            local_scores = np.full((n_bbo,), self.logzero)
            _best_odims = local_best_odims.view(n_bb, beam) + pad_o
            _best_odims = _best_odims.view(-1).cpu().numpy()
            _best_score = local_best_scores.view(-1).cpu().detach().numpy()
            local_scores[_best_odims] = _best_score
            local_scores = to_device(self, torch.from_numpy(local_scores).float()).view(batch, beam, self.odim) # 这里应该是所有的可能得分[batch,beam,odim]

            eos_vscores = local_scores[:, :, self.eos] + vscores # 关于eos这个分类的得分情况
            vscores = vscores.view(batch, beam, 1).repeat(1, 1, self.odim) # [batch,beam,1] -> [batch,beam,52] -> 感觉像是开辟空间
            vscores[:, :, self.eos] = self.logzero
            vscores = (vscores + local_scores).view(batch, n_bo) # 这里是[1,beam*odim]

            # global pruning (pad_b is 0 when batch=1)
            accum_best_scores, accum_best_ids = torch.topk(vscores, beam, 1) # prune and choose 在整个beam*odim的范围中进行结果挑选
            accum_odim_ids = torch.fmod(accum_best_ids, self.odim).view(-1).data.cpu().tolist() # 取余：查看每一个batch选的是哪一个输出字符 [wi]
            accum_padded_odim_ids = (torch.fmod(accum_best_ids, n_bo) + pad_bo).view(-1).data.cpu().tolist() # 筛选出哪一个char 这里实在整除            
            accum_padded_beam_ids  = (torch.div(accum_best_ids, self.odim) + pad_b).view(-1).data.cpu().tolist() # 筛选出哪一个Beam
            accum_padded_odim_ids1 = (torch.fmod(accum_best_ids, self.odim) + pad_b).view(-1).data.cpu().tolist() # 筛选出哪一个Beam(额外添加的，并不确定这里是否正确)

            y_prev = yseq[:][:]
            yseq = index_select_list(yseq, accum_padded_beam_ids) # 候选解码的所有序列
            yseq = append_ids(yseq, accum_odim_ids) # 添加挑选出来的label
            vscores = accum_best_scores # 活下来的beam的得分情况
            vidx = to_device(self, torch.LongTensor(accum_padded_beam_ids)) # Final idx[为什么只要保留batch的index]
            oidx = to_device(self, torch.LongTensor(accum_padded_odim_ids))

            ## 注意挑选1.postr_alpha需要两维挑选 2.prior_att_w需要进行一维的挑选 3.h2的挑选是进行一维的挑选 4.还有输出结果  5.还有对应的score情况  总共应该有五项结果 ##
            ## torch.index_select(target, dim, indices) 在哪一个维度上进行结果的索引的挑选
            ## postr_att_w[n_bb, T, odim] / prior_att_w[n_bb, T] 
            # 这里vidx是beam_idx的结果/这里
            prev_alpha = torch.index_select(prior_att_w.view(n_bb, *prior_att_w.shape[1:]), 0, vidx) # 挑选n_bb(beam*batch)的结果 挑选中间结果(1): prior_alpha
            h_ = torch.index_select(h2.view(n_bb, *h2.shape[1:]), 0, vidx) # 挑选所要的中间结果(3): state h2(30,800)
            # postr_vidx = to_device(self, torch.LongTensor(accum_padded_odim_ids))
            # tmps_postr = ( postr_att_w.transpose(0, 1).contiguous().view(-1,n_bo) ).transpose(0, 1)
            # prev_alpha  = torch.index_select(tmps_postr, 0, postr_vidx)            
            prev_postr = []
            for idx, [beamidx, charidx] in enumerate(zip(accum_padded_beam_ids, accum_padded_odim_ids1)):
                prev_postr.append( postr_att_w[beamidx,:,charidx] ) 
            prev_alpha = torch.stack(prev_postr, dim=0)
            ## for forward attention with TA
            prev_output = []
            for beamidx in accum_padded_beam_ids:
                prev_output.append( next_probs[beamidx,:] ) 
            next_probs = torch.stack(prev_output, dim=0)

            ## RNNLM Score
            if rnnlm:
                rnnlm_prev = index_select_lm_state(rnnlm_state, 0, vidx)
            ## CTC Score
            if lpz is not None:
                ctc_vidx = to_device(self, torch.LongTensor(accum_padded_odim_ids))
                ctc_scores_prev = torch.index_select(ctc_scores.view(-1), 0, ctc_vidx)
                ctc_scores_prev = ctc_scores_prev.view(-1, 1).repeat(1, self.odim).view(batch, n_bo)

                ctc_states = torch.transpose(ctc_states, 1, 3).contiguous()
                ctc_states = ctc_states.view(n_bbo, 2, -1)
                ctc_states_prev = torch.index_select(ctc_states, 0, ctc_vidx).view(n_bb, 2, -1)
                ctc_states_prev = torch.transpose(ctc_states_prev, 1, 2)

            # pick ended hyps
            if i > minlen:
                k = 0
                penalty_i = (i + 1) * penalty
                thr = accum_best_scores[:, -1]
                for samp_i in six.moves.range(batch): # 对于其中每一个batch进行修改
                    if stop_search[samp_i]:
                        k = k + beam
                        continue
                    for beam_j in six.moves.range(beam):
                        if eos_vscores[samp_i, beam_j] > thr[samp_i]:
                            yk = y_prev[k][:]
                            yk.append(self.eos)
                            if len(yk) < hlens[samp_i]:
                                _vscore = eos_vscores[samp_i][beam_j] + penalty_i
                                if normalize_score:
                                    _vscore = _vscore / len(yk)
                                _score = _vscore.data.cpu().numpy()
                                ended_hyps[samp_i].append({'yseq': yk, 'vscore': _vscore, 'score': _score})
                        k = k + 1

            # end detection
            stop_search = [stop_search[samp_i] or end_detect(ended_hyps[samp_i], i)
                           for samp_i in six.moves.range(batch)]
            stop_search_summary = list(set(stop_search))
            if len(stop_search_summary) == 1 and stop_search_summary[0]:
                break

            torch.cuda.empty_cache()

        dummy_hyps = [{'yseq': [self.sos, self.eos], 'score': np.array([-float('inf')])}]
        ended_hyps = [ended_hyps[samp_i] if len(ended_hyps[samp_i]) != 0 else dummy_hyps
                      for samp_i in six.moves.range(batch)]
        nbest_hyps = [sorted(ended_hyps[samp_i], key=lambda x: x['score'], reverse=True)[:min(len(ended_hyps[samp_i]), recog_args.nbest)]
                      for samp_i in six.moves.range(batch)]
        return nbest_hyps

    
    def calculate_all_attentions(self, hs_pad, hlens, ys_pad, strm_idx=0):  #### change here
        """Calculate all of attentions
            :param torch.Tensor hs_pad: batch of padded hidden state sequences (B, Tmax, D)
            :param torch.Tensor hlen: batch of lengths of hidden state sequences (B)
            :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
            :param int strm_idx: stream index for parallel speaker attention in multi-speaker case
            :return: attention weights with the following shape,
                1) multi-head case => attention weights (B, H, Lmax, Tmax),
                2) other case => attention weights (B, Lmax, Tmax).
            :rtype: float ndarray
        """
        # TODO(kan-bayashi): need to make more smart way
        ys = [y[y != self.ignore_id] for y in ys_pad]  # parse padded ys
        # attention index for the attention module in SPA (speaker parallel attention), att_idx is used to select attention module. In other cases, it is 0.
        att_idx = min(strm_idx, len(self.att) - 1) # useful only for SAT

        # hlen should be list of integer
        hlens = list(map(int, hlens))

        self.loss = None
        # prepare input and output word sequences with sos/eos IDs
        eos = ys[0].new([self.eos])
        sos = ys[0].new([self.sos])
        ys_in = [torch.cat([sos, y], dim=0) for y in ys] # in this way : the label and the feedback y is created (ys_in is for state(t-1))
        ys_out = [torch.cat([y, eos], dim=0) for y in ys] #  

        ## 先生成deocder_len的序列
        ys_out_len = [y.size(0) for y in ys_out]
        ymask = (to_device(self, make_trueframe_mask(ys_out_len))).float()
        context_mask = (to_device(self, make_trueframe_mask(hlens))).float() # [B,T]=[10,203]

        # padding for ys with -1
        # pys: utt x olen
        ys_in_pad = pad_list(ys_in, self.eos) # pad feature
        ys_out_pad = pad_list(ys_out, self.ignore_id) # pad output symbol
        ys_true_pad = pad_list(ys_out, self.eos) # pad output symbol

        # get dim, length info [B,T_dec] = [15,168]
        batch = ys_out_pad.size(0)
        olength = ys_out_pad.size(1)
        logging.info(self.__class__.__name__ + ' input lengths:  ' + str(hlens))
        logging.info(self.__class__.__name__ + ' output lengths: ' + str([y.size(0) for y in ys_out])) # verbose 0

        # initialization : give initial context_vector & decoder_state
        h_ = self.mean_init_state(hs_pad, hlens) # self.zero_state(hs_pad)
        # prev_alpha = None # previous prior attention score
        prev_postr_energy = (to_device(self, make_trueframe_mask(hlens))).float() # previous postr attention score
        prev_alpha = prev_postr_energy / (prev_postr_energy.sum(1, keepdim=True)+1e-8) # [B,T]
        next_probs = (1.0/self.odim)*torch.ones((batch, self.odim)).to(prev_alpha)

        self.att[att_idx].reset()  # reset pre-computation of h

        # pre-computation of embedding y -> emb(y) [B,C,D_senone]
        eys = self.dropout_emb(self.embed(ys_in_pad))  # utt x olen x zdim [B,decoder_T,300]

        # Posterior Attention&Decoder Process
        prior_att_weights = []
        # emb: [B,decoder_T, embsize]
        state_below = eys                # 处理后的y(t-1): [B,decoder_T,embsize]=[B,144,256]
        state_below_ = self.rnn1_W(eys)  # [B,decoder_T,dunits]=[10,144,600]
        state_belowx = self.rnn1_Wx(eys) # [B,decoder_T,dunits]=[10,144,300]

        if xinput_operation:
            enc_out_transform = self.classif_encoder_W(hs_pad) # [B,T,D]
            hs_pad_input = hs_pad.transpose(2,1).unsqueeze(2) # [B,T,D]->[B,D,T]->[B,D,1,T] 
            x_conv = self.loc_conv( hs_pad_input ).squeeze(2).transpose(2,1) # [B,D,1,T]->[B,D,T]->[B,T,D]
            enc_out_transform = enc_out_transform * torch.sigmoid(x_conv)            
        else:
            enc_out_transform = self.classif_encoder_W(hs_pad)
        for i in six.moves.range(olength):  # For every characters
            m_ = ymask[:,i] # 解码序列的MASK the variables for this char decoder steps
            x_ = state_below_[:,i,:]
            xx_ = state_belowx[:,i,:]
            cc_ = hs_pad

            ## 1. RNN eqn.(22): s'(t)=RNN(y(t-1),s(t-1)) || to combine [emb(t-1),state(t-1)] -> predt-state(t) || RNN(dunits) : concat[ey,ctx-previous-posterior]
            prev_context_vector = torch.sum(prev_alpha.unsqueeze(2)*cc_, dim=1)
            preact1 = self.rnn1_U(h_) # no bias -> [B,dunits*2]=[10,600]
            preact1 += self.rnn1_context_W( prev_context_vector ) # [B,T,eprojs][eprojs,dunits*2]
            preact1 += x_ # [B,dunits*2]=[10,600]
            preact1 = torch.sigmoid(preact1)
            r1 = self._slice(preact1, 0, self.dunits) # reset gate [B,dunits]=[10,300]
            u1 = self._slice(preact1, 1, self.dunits) # update gate [B,dunits]=[10,300]
            preactx1 = self.rnn1_Ux(h_) # no bias [B,dunits]=[10,300]
            preactx1 *= r1
            preactx1 += xx_ 
            preactx1 += self.rnn1_context_Wx( prev_context_vector ) 
            h1 = torch.tanh(preactx1)
            h1 = u1 * h_ + (1. - u1) * h1
            h1 = m_.unsqueeze(1) * h1 + (1. - m_).unsqueeze(1) * h_

            ## 2. Attention eqn.(23): c(t)=AttentionContext(s'(t), enc_o) || dunits (prior_att_c[B,eprojs] prior_att_w[B,T])
            prior_att_c, prior_att_w = self.att[att_idx](cc_, hlens, h1, prev_alpha) 
            prior_att_weights.append(prior_att_w)
            
            ## 3. RNN eqn.(24): s(t)=RNN(c(t),s'(t)) || att_c(eprojs) z_infocomb_list(dunits)
            preact2 = self.rnn2_U_n1(h1) # [B,dunits*2]=[10,600]
            preact2 += self.rnn2_Wc(prior_att_c) # no bias [B,dunits*2]=[10,600]
            preact2 = torch.sigmoid(preact2)
            r2 = self._slice(preact2, 0, self.dunits) # [B,dunits]
            u2 = self._slice(preact2, 1, self.dunits) # [B,dunits]
            preactx2 = self.rnn2_Ux_n1(h1) # [B,dunits]=[10,300]
            preactx2 *= r2
            preactx2 += self.rnn2_Wcx(prior_att_c) # [B,dunits]=[10,300]
            h2 = torch.tanh(preactx2)
            h2 = u2 * h1 + (1. - u2) * h2
            h2 = m_.unsqueeze(1) * h2 + (1. - m_).unsqueeze(1) * h1 # [B,dunits]=[10,300]

            ## 4. Classification for every frames
            logit = self.classif_state_W(h2) + self.classif_y_prev_W(state_below[:, i, :]) # [B,dunits]=[10,300]
            logit_all_for_classif = logit.unsqueeze(1) + enc_out_transform # [B,T,dunits]=[10,203,300]
            
            nframes = logit_all_for_classif.size(1) # T 203
            probs = logit_all_for_classif # [B,T,52]            

            probs = torch.reshape(probs, (batch*nframes, self.odim))
            probs = F.softmax(probs, dim=-1) 
            probs = torch.reshape(probs, (batch, nframes, self.odim)) # [B,T,odim]=[10,203,52]
            next_probs = torch.sum(probs * (prior_att_w.unsqueeze(2) + 1e-8), dim=1) # [B,T,odim]->[B,52]
            # y_argmax_index = torch.argmax(next_probs, dim=1) # [B]=[10]

            ## 5.create postr attention scores(修正：尝试老的实现方式)
            minitiny = np.finfo('float32').tiny
            choose_topK = 1
            next_probs_argsort = torch.argsort(next_probs, dim=1, descending=False) # default is 0 -> 1
            topK_index = (next_probs_argsort[:,-1*choose_topK:]).long() # [B,odim]->[B]
            for_index_add = torch.arange(batch).long().to(topK_index)*self.odim
            topK_index = topK_index + for_index_add[:,None] # [B,topK]=[B]
            saved_index = topK_index.reshape([batch, choose_topK]) # [B,topK]
            padding = ((saved_index[:,0]).unsqueeze(1).repeat(1, self.odim-choose_topK)).long() # [B,odim-topK]
            all_index = torch.cat( [saved_index, padding], dim=1 ).long() # [B,odim]
            all_index = all_index.reshape([batch*self.odim])
            get_mask = torch.zeros((batch*self.odim)).to(next_probs)
            get_mask = get_mask.index_fill_(0, all_index, 1)
            get_mask = torch.reshape(get_mask, (batch, self.odim)) # [B,self.odim]
            masked_probs_energy = torch.exp( \
                torch.log(torch.clamp(probs * get_mask.unsqueeze(1), minitiny, 1.)) + \
                torch.log(torch.clamp(prior_att_w.unsqueeze(2),      minitiny, 1.)) \
                ) # [B,T,odim]*[B,None,odim]*[B,T,None]
            masked_probs_energy = torch.clamp(masked_probs_energy, minitiny, 1.)
            masked_probs_energy = torch.sum(masked_probs_energy, dim=2) # [B,T]
            masked_probs_energy = masked_probs_energy*context_mask
            postr_alpha = masked_probs_energy / (minitiny + masked_probs_energy.sum(1, keepdim=True)) # True posterior attention part (B,T)           

            ## 6.create postr attention scores
            true_mask = slow_onehot(labels=ys_true_pad[:,i], senone=self.odim)
            masked_probs_energy = torch.exp( \
                torch.log(torch.clamp(probs * true_mask.unsqueeze(1), 1e-8, 1.)) + \
                torch.log(torch.clamp(prior_att_w.unsqueeze(2), 1e-8, 1.)) \
                ) # [B,T,odim]
            masked_probs_energy = torch.clamp(masked_probs_energy, 1e-8, 1.)
            masked_probs_energy = torch.sum(masked_probs_energy, dim=2) # [B,T]
            masked_probs_energy = masked_probs_energy*context_mask
            postr_alpha_true = masked_probs_energy / (1e-8 + masked_probs_energy.sum(1, keepdim=True)) # True posterior attention part (B,T)    

            ## 7. pred posterior attention score
            distance = kullback_leibler_divergence(postr_alpha_true, postr_alpha, axis=1) # (B,T)
            distance = torch.where( distance <= 1e-8, torch.zeros(distance.size()).to(distance), distance)            

            ## 8. update previous variables and collect all y_predict/postr_ytrue/postr_pred
            prev_alpha = postr_alpha_true
            h_ = h2

        ## 1. convert to numpy array with the shape (B, Lmax, Tmax)
        prior_score = att_to_numpy(prior_att_weights, self.att[att_idx])
        return prior_score

        # postr_pred_score = att_to_numpy(postr_pred_att_weights, self.att[att_idx])
        # postr_true_score = att_to_numpy(postr_true_att_weights, self.att[att_idx])
        # final_att_ws = np.concatenate( (prior_score[:,None,:,:], np.abs(postr_pred_score-postr_true_score)[:,None,:,:], postr_true_score[:,None,:,:], postr_pred_score[:,None,:,:]), axis=1)

        # ## 2. show some other intermediate variables
        # intermediate_var = att_to_numpy(intermediate_var, self.att[att_idx])
        # probs_var = att_to_numpy(probs_var, self.att[att_idx])
        # pmask_probs_var = att_to_numpy(pmask_probs_var, self.att[att_idx])
        # tmask_probs_var = att_to_numpy(tmask_probs_var, self.att[att_idx])
        # final_var = np.concatenate( (intermediate_var[:,None,:,:,:], probs_var[:,None,:,:,:], pmask_probs_var[:,None,:,:,:], tmask_probs_var[:,None,:,:,:]), axis=1)
        
        # ## 3. classification three parts
        # enc_out_w = enc_out_transform.detach().cpu().numpy() # [B,T,dunits]
        # state_var = att_to_numpy(state_var, self.att[att_idx]) # [B,decoderT,dunits]
        # ytemb_var = att_to_numpy(ytemb_var, self.att[att_idx]) # [B,decoderT,dunits]
        # s_y_var = np.concatenate( (state_var[:,None,:,:], ytemb_var[:,None,:,:]), axis=1)

        # ## 4. mask variables
        # pred_mask_var = att_to_numpy(pred_mask_var, self.att[att_idx]) # [B,decT,odim]
        # true_mask_var = att_to_numpy(true_mask_var, self.att[att_idx])
        # mask_var = np.concatenate( (pred_mask_var[:,None,:,:], true_mask_var[:,None,:,:]), axis=1) # [B,DecT,odim]

        # return [final_att_ws, final_var, enc_out_w, s_y_var, mask_var]



def decoder_postrtf_for(args, odim, sos, eos, att, labeldist): # decoder_postrv2cmtrue_for
    return IntergrationPostrPriorDecoder(args.eprojs, odim, args.dtype, args.dlayers, args.dunits, sos, eos, att, args.verbose,
                   args.char_list, labeldist,
                   args.lsm_weight, args.sampling_probability, args.dropout_rate_decoder, 
                   dec_filts=args.dec_filts)
