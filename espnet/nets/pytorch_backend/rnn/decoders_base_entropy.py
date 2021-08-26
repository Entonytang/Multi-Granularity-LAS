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
MAX_DECODER_OUTPUT = 5 # ?
CTC_SCORING_RATIO = 1.5 # ?
traditional_loss_calculate = False

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


class SoftmaxDecoder(torch.nn.Module):
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
                    verbose=0, char_list=None, labeldist=None, lsm_weight=0., sampling_probability=0.0, dropout=0.0, embsize=256):
        super(SoftmaxDecoder, self).__init__()
        self.dtype = dtype
        self.dunits = dunits
        self.dlayers = dlayers
        self.embsize = embsize
        self.embed = torch.nn.Embedding(odim, self.embsize)
        self.dropout_emb = torch.nn.Dropout(p=dropout)
        
        ## Split RNN Information Combination Layer ##
        self.rnn1_Wx = torch.nn.Linear(embsize, dunits)
        self.rnn1_W = torch.nn.Linear(embsize, dunits*2)
        self.rnn1_U = torch.nn.Linear(dunits, dunits*2, bias=False)
        self.rnn1_Ux = torch.nn.Linear(dunits, dunits, bias=False)
        ## Split RNN for next iterator layer ##
        self.rnn2_U_n1 = torch.nn.Linear(dunits, dunits*2)
        self.rnn2_Wc = torch.nn.Linear(eprojs, dunits*2, bias=False)
        self.rnn2_Ux_n1 = torch.nn.Linear(dunits, dunits)
        self.rnn2_Wcx = torch.nn.Linear(eprojs, dunits, bias=False)
        ## New added here by jtang ##
        self.init_state_fc = torch.nn.Linear(eprojs, dunits)
        ## Classification Part ##
        self.classif_state_W = torch.nn.Linear(dunits, dunits)
        self.classif_y_prev_W = torch.nn.Linear(self.embsize, dunits, bias=False)
        self.classif_encoder_W = torch.nn.Linear(eprojs, dunits, bias=False)
        self.classif_logit_W = torch.nn.Linear(dunits//2, odim)
        print('Setting loss calc', traditional_loss_calculate)
        print('This is only decoder base type. (base dec)')
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


    # Not Finished (not check). 代码应该是还有问题的：需要进行相应的修改
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
        probs = (1/self.odim)*torch.ones((batch, self.odim)).to(prev_alpha)

        self.att[att_idx].reset()  # reset pre-computation of h

        # pre-computation of embedding y -> emb(y) [B,C,D_senone]
        eys = self.dropout_emb(self.embed(ys_in_pad))  # utt x olen x zdim [B,decoder_T,300]

        # Posterior Attention&Decoder Process
        prior_att_weights = []
        prior_att_context = []
        output_all = []
        
        # 修改：这里的输入是emb：[B,decoder_T, embsize]
        state_below = eys                # 处理后的y(t-1): [B,decoder_T,embsize]=[B,144,256]
        state_below_ = self.rnn1_W(eys)  # [B,decoder_T,dunits]=[10,144,600]
        state_belowx = self.rnn1_Wx(eys) # [B,decoder_T,dunits]=[10,144,300]
        for i in six.moves.range(olength):  # For every characters
            m_ = ymask[:,i] # 解码序列的MASK the variables for this char decoder steps
            x_ = state_below_[:,i,:]
            xx_ = state_belowx[:,i,:]
            cc_ = hs_pad

            ## 1. RNN eqn.(22): s'(t)=RNN(y(t-1),s(t-1)) || to combine [emb(t-1),state(t-1)] -> predt-state(t) || RNN(dunits) : concat[ey,ctx-previous-posterior]
            preact1 = self.rnn1_U(h_) # no bias -> [B,dunits*2]=[10,600]
            preact1 += x_ # [B,dunits*2]=[10,600]
            preact1 = torch.sigmoid(preact1)
            r1 = self._slice(preact1, 0, self.dunits) # reset gate [B,dunits]=[10,300]
            u1 = self._slice(preact1, 1, self.dunits) # update gate [B,dunits]=[10,300]
            preactx1 = self.rnn1_Ux(h_) # no bias [B,dunits]=[10,300]
            preactx1 *= r1
            preactx1 += xx_ 
            h1 = torch.tanh(preactx1)
            h1 = u1 * h_ + (1. - u1) * h1
            h1 = m_.unsqueeze(1) * h1 + (1. - m_).unsqueeze(1) * h_

            ## 2. Attention eqn.(23): c(t)=AttentionContext(s'(t), enc_o) || dunits (prior_att_c[B,eprojs] prior_att_w[B,T])
            prior_att_c, prior_att_w = self.att[att_idx](cc_, hlens, h1, prev_alpha) 
            prior_att_weights.append(prior_att_w)
            prior_att_context.append(prior_att_c)
            
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
            logit_all_for_classif = self.classif_state_W(h2) + self.classif_y_prev_W(state_below[:, i, :]) + self.classif_encoder_W(prior_att_c) # [B,dunits]=[10,300]
            logit_all_for_classif    = torch.reshape( logit_all_for_classif, (batch, self.dunits//2, 2) )
            logit_all_for_classif, _ = torch.max( logit_all_for_classif, dim=-1 ) # [B,dunits//2]=[B,150]
            probs = self.classif_logit_W(logit_all_for_classif) # [B,odim]=[10,52]
            probs = F.softmax(probs, dim=-1) # [B,52]
            
            ## 5. save posterior output
            output_all.append(probs)
            
            ## 6. update previous variables
            h_ = h2
            prev_alpha = prior_att_w

        ## 8. Get mainly loss : CE and auxiliary loss : KL-penalty
        y_predict_all   = torch.stack(output_all, dim=1).view(batch * olength, self.odim)   # [B,decoder_T, odim]->[B*decoder_T, odim]
        if traditional_loss_calculate == True:
            reduction_str = 'elementwise_mean' if LooseVersion(torch.__version__) < LooseVersion('1.0') else 'mean'
            # input : [N,C] label : [N] softmax is in F.cross_entropy (o:scalar)
            self.loss = F.categorical_crossentropy(y_predict_all, ys_out_pad.view(-1), ignore_index=self.ignore_id, reduction=reduction_str)
            # -1: eos, which is removed in the loss computation
            self.loss *= (np.mean([len(x) for x in ys_in]) - 1)
        else:
            self.loss = F.categorical_crossentropy(y_predict_all, ys_out_pad.contiguous().view(-1).long(), ignore_index=self.ignore_id, reduction='none')
            self.loss = self.loss.view(batch, olength) 
            self.loss = self.loss.sum(1)
            self.loss = self.loss.mean(0) # to be scalar 对比过theano的处理：和theano已经大体一致 torch:33.8797/theano=33.88
        # pad_outputs: prediction tensors (B*decoder_T, D)
        # pad_targets: target tensors (B, decoder_T, D)
        acc = th_accuracy(y_predict_all, ys_out_pad, ignore_label=self.ignore_id)
        logging.info('att loss:' + ''.join(str(self.loss.item()).split('\n')))

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

        if self.labeldist is not None: # 这里应该使用不到的吧()
            if self.vlabeldist is None:
                self.vlabeldist = to_device(self, torch.from_numpy(self.labeldist))
            # loss_reg = - torch.sum((F.log_softmax(y_predict_all, dim=1) * self.vlabeldist).view(-1), dim=0) / len(ys_in)
            loss_reg = - torch.sum((torch.log(y_predict_all) * self.vlabeldist).view(-1), dim=0) / len(ys_in)
            self.loss = (1. - self.lsm_weight) * self.loss + self.lsm_weight * loss_reg

        auxilitary_entropy_loss = True
        if auxilitary_entropy_loss == True:
            ## calcualte entropy of attention score start  ##
            bpe_attention_score = torch.stack(prior_att_weights, dim=1) # BPE alpha score
            bpe_entropy = torch.sum(-1 * bpe_attention_score * torch.log(bpe_attention_score + 1e-10), dim=-1) # (B,decT,T)->[B.decT]
            bpelens = torch.tensor(ys_out_len).to(bpe_attention_score)
            ent_bpe = ((1.0/(bpelens) * (bpe_entropy).sum(1)).to(bpe_attention_score)).mean(0)
            ## calcualte entropy of attention score finish ##

        # print(ent_bpe)
        # exit(0)

        return self.loss, acc, torch.zeros(self.loss.size()), ent_bpe

    # Not Finished (not check). this code perherps useless
    def recognize_beam(self, h, lpz, recog_args, char_list, rnnlm=None, strm_idx=0): # h(enc_o) [T,D_enc] lpz(CTC output)
        """beam search implementation

            :param torch.Tensor h: encoder hidden state (T, eprojs)
            :param torch.Tensor lpz: ctc log softmax output (T, odim)
            :param Namespace recog_args: argument Namespace containing options
            :param char_list: list of character strings
            :param torch.nn.Module rnnlm: language module
            :param int strm_idx: stream index for speaker parallel attention in multi-speaker case (SAT)
            :return: N-best decoding results
            :rtype: list of dicts
        """
        logging.info('input lengths: ' + str(h.size(0)))
        att_idx = min(strm_idx, len(self.att) - 1)
        # initialization
        c_list = [self.zero_state(h.unsqueeze(0))]
        z_list = [self.zero_state(h.unsqueeze(0))]
        for _ in six.moves.range(1, self.dlayers):
            c_list.append(self.zero_state(h.unsqueeze(0)))
            z_list.append(self.zero_state(h.unsqueeze(0)))
        ## New added here by jtang ##
        c_infocomb_list = self.zero_state(h.unsqueeze(0)) # offer batch dimension to be 1
        z_infocomb_list = self.mean_init_state_decoder(h)
        #############################
        a = None
        self.att[att_idx].reset()  # reset pre-computation of h

        # search parms
        beam = recog_args.beam_size
        ctc_weight = recog_args.ctc_weight

        # preprate sos
        y = self.sos
        vy = h.new_zeros(1).long()

        if recog_args.maxlenratio == 0:
            maxlen = h.shape[0]
        else:
            maxlen = max(1, int(recog_args.maxlenratio * h.size(0))) # maxlen >= 1
        minlen = int(recog_args.minlenratio * h.size(0))
        logging.info('max output length: ' + str(maxlen))
        logging.info('min output length: ' + str(minlen))

        # initialize hypothesis
        if rnnlm:
            hyp = {'score': 0.0, 'yseq': [y], 'c_prev': c_list, 'z_prev': z_list, 'ci_prev': c_infocomb_list, 'zi_prev': z_infocomb_list, 'a_prev': a, 'rnnlm_prev': None}
        else:
            hyp = {'score': 0.0, 'yseq': [y], 'c_prev': c_list, 'z_prev': z_list, 'ci_prev': c_infocomb_list, 'zi_prev': z_infocomb_list, 'a_prev': a}
        if lpz is not None:
            ctc_prefix_score = CTCPrefixScore(lpz.detach().numpy(), 0, self.eos, np)
            hyp['ctc_state_prev'] = ctc_prefix_score.initial_state() # 
            hyp['ctc_score_prev'] = 0.0
            if ctc_weight != 1.0:
                # pre-pruning based on attention scores
                ctc_beam = min(lpz.shape[-1], int(beam * CTC_SCORING_RATIO)) # 
            else:
                ctc_beam = lpz.shape[-1]
        hyps = [hyp]
        ended_hyps = []

        for i in six.moves.range(maxlen):
            logging.debug('position ' + str(i))

            hyps_best_kept = []
            for hyp in hyps:
                vy.unsqueeze(1)
                vy[0] = hyp['yseq'][i]
                ey = self.dropout_emb(self.embed(vy))                                                                                                               # utt list (1) x zdim : ey[B,D_emb]
                ey.unsqueeze(0)                                                                                                                                     # [1,D_emb]
                ## Counterpart to SS : but no SS in decoding
                ## 1. RNN eqn.(22): s'(t)=RNN(y(t-1),s(t-1)) || to combine [emb(t-1),state(t-1)] -> predt-state(t) || RNN(dunits) 
                z_infocomb_list, c_infocomb_list = self.rnn_forward_one(ey, hyp['zi_prev'], hyp['ci_prev'])                                                         # z_infocomb_list(output) like the h1 while z_infocomb_list(input) is h_
                ## 2. Attention eqn.(23): c(t)=AttentionContext(s'(t), enc_o) || dunits
                att_c, att_w = self.att[att_idx](h.unsqueeze(0), [h.size(0)], self.dropout_dec[0](z_infocomb_list), hyp['a_prev'])                                  # atten(enc_o, enc_len, prev_y, prev_alpha)
                ## 3. RNN eqn.(24): s(t)=RNN(c(t),s'(t)) || att_c(eprojs) z_infocomb_list(dunits)
                z_list[0] = z_infocomb_list
                z_list, c_list = self.rnn_forward(att_c, z_list, c_list, z_list, c_list)                                                                            # (y_emb(t):) do rnn part : z_list -> output[h2]
                ## 4. Offer s(t) for classification operation
                z_infocomb_list = z_list[-1]                                                                                                                        # s(t) [B,dunits]
                ## 5. Compute loss( ff_logit_lstm[state(t)] + ff_logit_prev[emb(t-1)] + ff_logit_ctx(ctx(t)) )
                all_information_combine = self.ff_logit_lstm_fc(z_infocomb_list) + self.ff_logit_emb_fc(ey) + self.ff_logit_ctx_fc(att_c)                           # [B,dunits] [B,dunits] [B,eprojs] 
                y_all = self.output(all_information_combine)                                                                                                        # do final FC -> transform it to 52 & softmax

                # get nbest local scores and their ids
                local_att_scores = F.log_softmax(y_all, dim=1)                                                                                                      # + log for score fusion
                if rnnlm:
                    rnnlm_state, local_lm_scores = rnnlm.predict(hyp['rnnlm_prev'], vy)
                    local_scores = local_att_scores + recog_args.lm_weight * local_lm_scores
                else:
                    local_scores = local_att_scores

                if lpz is not None:
                    local_best_scores, local_best_ids = torch.topk( local_att_scores, ctc_beam, dim=1)                                                              # topk for beam-search
                    ctc_scores, ctc_states = ctc_prefix_score( hyp['yseq'], local_best_ids[0], hyp['ctc_state_prev'] )
                    local_scores = (1.0 - ctc_weight) * local_att_scores[:, local_best_ids[0]] + ctc_weight * torch.from_numpy(ctc_scores - hyp['ctc_score_prev'])  # all score
                    if rnnlm:
                        local_scores += recog_args.lm_weight * local_lm_scores[:, local_best_ids[0]]                                                                # add rnnlm
                    local_best_scores, joint_best_ids = torch.topk(local_scores, beam, dim=1)                                                                       # Final Score
                    local_best_ids = local_best_ids[:, joint_best_ids[0]]
                else:
                    local_best_scores, local_best_ids = torch.topk(local_scores, beam, dim=1)

                for j in six.moves.range(beam):
                    new_hyp = {}
                    # [:] is needed! only score/yseq is useful.
                    new_hyp['z_prev'] = z_list[:]                                                   # I don't think its useful.
                    new_hyp['c_prev'] = c_list[:]                                                   # I don't think its useful.
                    new_hyp['a_prev'] = att_w[:]                                                    # I don't think its useful.
                    new_hyp['score'] = hyp['score'] + local_best_scores[0, j]
                    new_hyp['yseq'] = [0] * (1 + len(hyp['yseq']))
                    new_hyp['yseq'][:len(hyp['yseq'])] = hyp['yseq']
                    new_hyp['yseq'][len(hyp['yseq'])] = int(local_best_ids[0, j])
                    if rnnlm:
                        new_hyp['rnnlm_prev'] = rnnlm_state
                    if lpz is not None:
                        new_hyp['ctc_state_prev'] = ctc_states[joint_best_ids[0, j]]
                        new_hyp['ctc_score_prev'] = ctc_scores[joint_best_ids[0, j]]
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)
                hyps_best_kept = sorted(hyps_best_kept, key=lambda x: x['score'], reverse=True)[:beam]

            # sort and get nbest
            hyps = hyps_best_kept
            logging.debug('number of pruned hypotheses: ' + str(len(hyps)))
            logging.debug('best hypo: ' + ''.join([char_list[int(x)] for x in hyps[0]['yseq'][1:]]))

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                logging.info('adding <eos> in the last position in the loop')
                for hyp in hyps:
                    hyp['yseq'].append(self.eos)

            # add ended hypotheses to a final list, and removed them from current hypotheses (this will be a problem, number of hyps < beam).
            remained_hyps = []
            for hyp in hyps:
                if hyp['yseq'][-1] == self.eos:
                    # only store the sequence that has more than minlen outputs also add penalty
                    if len(hyp['yseq']) > minlen:
                        hyp['score'] += (i + 1) * penalty
                        if rnnlm: # Word LM needs to add final <eos> score
                            hyp['score'] += recog_args.lm_weight * rnnlm.final(hyp['rnnlm_prev'])
                        ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            # end detection
            if end_detect(ended_hyps, i) and recog_args.maxlenratio == 0.0:
                logging.info('end detected at %d', i)
                break

            hyps = remained_hyps
            if len(hyps) > 0:
                logging.debug('remaining hypotheses: ' + str(len(hyps)))
            else:
                logging.info('no hypothesis. Finish decoding.')
                break

            for hyp in hyps:
                logging.debug('hypo: ' + ''.join([char_list[int(x)] for x in hyp['yseq'][1:]]))

            logging.debug('number of ended hypotheses: ' + str(len(ended_hyps)))

        nbest_hyps = sorted(ended_hyps, key=lambda x: x['score'], reverse=True)[:min(len(ended_hyps), recog_args.nbest)]
        # check number of hypotheses
        if len(nbest_hyps) == 0:
            logging.warning('there is no N-best results, perform recognition again with smaller minlenratio.')
            # should copy because Namespace will be overwritten globally
            recog_args = Namespace(**vars(recog_args))
            recog_args.minlenratio = max(0.0, recog_args.minlenratio - 0.1)
            return self.recognize_beam(h, lpz, recog_args, char_list, rnnlm)

        logging.info('total log probability: ' + str(nbest_hyps[0]['score']))
        logging.info('normalized log probability: ' + str(nbest_hyps[0]['score'] / len(nbest_hyps[0]['yseq'])))

        # remove sos
        return nbest_hyps

    # Not Finished (not check). 
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
        # mask 采用全1的初始化方式,因为batch=1恒成立,所以不必担心这一点

        if lpz is not None: # no need to modification
            device_id = torch.cuda.device_of(next(self.parameters()).data).idx
            ctc_prefix_score = CTCPrefixScoreTH(lpz, 0, self.eos, beam, exp_hlens, device_id)
            ctc_states_prev = ctc_prefix_score.initial_state()
            ctc_scores_prev = to_device(self, torch.zeros(batch, n_bo))

        prior_att_weights = []
        prior_att_context = []
        output_all = []

        for i in six.moves.range(maxlen):
            logging.debug('position ' + str(i))
            vy = to_device(self, torch.LongTensor(get_last_yseq(yseq))) # get the last char
            ey = self.dropout_emb(self.embed(vy)) # [n_bb, embsize]
            xx_ = self.rnn1_Wx(ey) # state_belowx[n_bb, dunits]
            x_ = self.rnn1_W(ey) # [n_bb, 2*dunits]
            cc_ = exp_h # [n_bb, T, eprojs]

            ## 1. RNN eqn.(22): s'(t)=RNN(y(t-1),s(t-1)) || to combine [emb(t-1),state(t-1)] -> predt-state(t) || RNN(dunits) : concat[ey,ctx-previous-posterior]
            preact1 = self.rnn1_U(h_) # [n_bb,dunits*2]
            preact1 += x_ # [n_bb,dunits*2]
            preact1 = torch.sigmoid(preact1)
            r1 = self._slice(preact1, 0, self.dunits) # reset gate [n_bb,dunits]
            u1 = self._slice(preact1, 1, self.dunits) # update gate [n_bb,dunits]
            preactx1 = self.rnn1_Ux(h_) # [n_bb,dunits]
            preactx1 *= r1
            preactx1 += xx_ 
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
            logit_all_for_classif = self.classif_state_W(h2) + self.classif_y_prev_W(ey) + self.classif_encoder_W(prior_att_c) # [n_bb,dunits]
            similarb = logit_all_for_classif.size(0) # 
            logit_all_for_classif    = torch.reshape( logit_all_for_classif, (similarb, self.dunits//2, 2) )
            logit_all_for_classif, _ = torch.max( logit_all_for_classif, dim=2 ) # [n_bb,dunits//2]
            probs = self.classif_logit_W(logit_all_for_classif) # [n_bb,odim]
            probs = F.softmax(probs, dim=1) # [n_bb,odim]

            # get nbest local scores and their ids                                                                                                              # 
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
            ## for forward attention with TA
            prev_output = []
            for beamidx in accum_padded_beam_ids:
                prev_output.append( probs[beamidx,:] ) 
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

    # Not Finished (not check).
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
        probs = (1/self.odim)*torch.ones((batch, self.odim)).to(prev_alpha)

        self.att[att_idx].reset()  # reset pre-computation of h

        # pre-computation of embedding y -> emb(y) [B,C,D_senone]
        eys = self.dropout_emb(self.embed(ys_in_pad))  # utt x olen x zdim [B,decoder_T,300]

        # Posterior Attention&Decoder Process
        prior_att_weights = []
        prior_att_context = []
        output_all = []
        
        # 修改：这里的输入是emb：[B,decoder_T, embsize]
        state_below = eys                # 处理后的y(t-1): [B,decoder_T,embsize]=[B,144,256]
        state_below_ = self.rnn1_W(eys)  # [B,decoder_T,dunits]=[10,144,600]
        state_belowx = self.rnn1_Wx(eys) # [B,decoder_T,dunits]=[10,144,300]
        for i in six.moves.range(olength):  # For every characters
            m_ = ymask[:,i] # 解码序列的MASK the variables for this char decoder steps
            x_ = state_below_[:,i,:]
            xx_ = state_belowx[:,i,:]
            cc_ = hs_pad

            ## 1. RNN eqn.(22): s'(t)=RNN(y(t-1),s(t-1)) || to combine [emb(t-1),state(t-1)] -> predt-state(t) || RNN(dunits) : concat[ey,ctx-previous-posterior]
            preact1 = self.rnn1_U(h_) # no bias -> [B,dunits*2]=[10,600]
            preact1 += x_ # [B,dunits*2]=[10,600]
            preact1 = torch.sigmoid(preact1)
            r1 = self._slice(preact1, 0, self.dunits) # reset gate [B,dunits]=[10,300]
            u1 = self._slice(preact1, 1, self.dunits) # update gate [B,dunits]=[10,300]
            preactx1 = self.rnn1_Ux(h_) # no bias [B,dunits]=[10,300]
            preactx1 *= r1
            preactx1 += xx_ 
            h1 = torch.tanh(preactx1)
            h1 = u1 * h_ + (1. - u1) * h1
            h1 = m_.unsqueeze(1) * h1 + (1. - m_).unsqueeze(1) * h_

            ## 2. Attention eqn.(23): c(t)=AttentionContext(s'(t), enc_o) || dunits (prior_att_c[B,eprojs] prior_att_w[B,T])
            prior_att_c, prior_att_w = self.att[att_idx](cc_, hlens, h1, prev_alpha) 
            prior_att_weights.append(prior_att_w)
            prior_att_context.append(prior_att_c)
            
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
            logit_all_for_classif = self.classif_state_W(h2) + self.classif_y_prev_W(state_below[:, i, :]) + self.classif_encoder_W(prior_att_c) # [B,dunits]=[10,300]
            logit_all_for_classif    = torch.reshape( logit_all_for_classif, (batch, self.dunits//2, 2) )
            logit_all_for_classif, _ = torch.max( logit_all_for_classif, dim=-1 ) # [B,dunits//2]=[B,150]
            probs = self.classif_logit_W(logit_all_for_classif) # [B,odim]=[10,52]
            probs = F.softmax(probs, dim=-1) # [B,52]
            
            ## 5. save posterior output
            output_all.append(probs)
            
            ## 6. update previous variables
            h_ = h2
            prev_alpha = prior_att_w
        
        att_ws = att_to_numpy(prior_att_weights, self.att[att_idx]) # change it : if [B, 4, DecT, T] order-Prior(1) Prior-Postr-abs(2) Postr-True(3) Postr-argmax(4)
        return att_ws


def decoder_base_entropy_for(args, odim, sos, eos, att, labeldist):
    return SoftmaxDecoder(args.eprojs, odim, args.dtype, args.dlayers, args.dunits, sos, eos, att, args.verbose,
                   args.char_list, labeldist,
                   args.lsm_weight, args.sampling_probability, args.dropout_rate_decoder)
