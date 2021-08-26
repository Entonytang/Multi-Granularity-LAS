#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""RNN sequence-to-sequence speech recognition model (pytorch)."""

from __future__ import division

import argparse
import logging
import math
import os
import copy
import editdistance

import chainer
import numpy as np
import six
import torch

from itertools import groupby

from chainer import reporter

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.e2e_asr_common import label_smoothing_dist
from espnet.nets.e2e_asr_common import label_smoothing_bpe_dist
from espnet.nets.pytorch_backend.ctc import ctc_for
from espnet.nets.pytorch_backend.initialization import lecun_normal_init_parameters
from espnet.nets.pytorch_backend.initialization import set_forget_bias_to_one
from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.pytorch_backend.nets_utils import to_torch_tensor
from espnet.nets.pytorch_backend.rnn.attentions import att_for
from espnet.nets.pytorch_backend.rnn.encoders import encoder_for
from espnet.nets.pytorch_backend.rnn.decoders         import decoder_for
from espnet.nets.pytorch_backend.rnn.decoders_mltask  import decoder_mtask_for
from espnet.nets.pytorch_backend.rnn.decoders_base    import decoder_base_for
from espnet.nets.pytorch_backend.rnn.decoders_mltaski import decoder_mtaski_for
from espnet.nets.pytorch_backend.rnn.decoders_mltaskr2 import decoder_mtaskr2_for
from espnet.nets.pytorch_backend.rnn.decoders_mltaskr import decoder_mtaskr_for

from espnet.nets.pytorch_backend.rnn.decoders_base_mltaskrllm3_ss import decoder_base_mltaskller_modification3_for
from espnet.nets.pytorch_backend.rnn.decoders_base_mltaskrll_modif3c_ss import decoder_base_mltaskller_m3c_schedulesampling_for
from espnet.nets.pytorch_backend.rnn.decoders_base_mltaskrll_modif3_ss2 import decoder_base_mltaskller_m3_schedulesampling2_for
from espnet.nets.pytorch_backend.rnn.decoders_base_mltaskrll_modif3_ss3 import decoder_base_mltaskller_m3_schedulesampling3_for

from espnet.nets.pytorch_backend.rnn.decoders_mltaskrll_m3ss import decoder_base_newmltaskr_m3ss_for #####

from espnet.nets.scorers.ctc import CTCPrefixScorer

CTC_LOSS_THRESHOLD = 10000


class MltReporter(chainer.Chain):
    """A chainer reporter wrapper"""

    def report(self, loss_ctc, loss_att_char, loss_att_bpe, acc_char, acc_bpe, cer_ctc, cer, wer, mtl_loss):
        reporter.report({'loss_ctc': loss_ctc}, self)
        reporter.report({'loss_att_char': loss_att_char}, self)
        reporter.report({'loss_att_bpe': loss_att_bpe}, self)
        reporter.report({'acc_bpe': acc_char}, self)
        reporter.report({'acc': acc_bpe}, self)
        reporter.report({'cer_ctc': cer_ctc}, self)
        reporter.report({'cer': cer}, self)
        reporter.report({'wer': wer}, self)
        logging.info('mtl loss:' + str(mtl_loss))
        reporter.report({'loss': mtl_loss}, self)


class E2Eschedules(ASRInterface, torch.nn.Module):
    """E2Eschedules module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments."""
        E2Eschedules.encoder_add_arguments(parser)
        E2Eschedules.attention_add_arguments(parser)
        E2Eschedules.decoder_add_arguments(parser)
        return parser

    @staticmethod
    def encoder_add_arguments(parser):
        """Add arguments for the encoder."""
        group = parser.add_argument_group("E2Eschedules encoder setting")
        # encoder
        group.add_argument('--etype', default='blstmp', type=str,
                           choices=['lstm', 'blstm', 'lstmp', 'blstmp', 'vgglstmp', 'vggblstmp', 'vgglstm', 'vggblstm',
                                    'gru', 'bgru', 'grup', 'bgrup', 'vgggrup', 'vggbgrup', 'vgggru', 'vggbgru'],
                           help='Type of encoder network architecture')
        group.add_argument('--elayers', default=4, type=int,
                           help='Number of encoder layers (for shared recognition part in multi-speaker asr mode)')
        group.add_argument('--eunits', '-u', default=300, type=int,
                           help='Number of encoder hidden units')
        group.add_argument('--eprojs', default=320, type=int,
                           help='Number of encoder projection units')
        group.add_argument('--subsample', default="1", type=str,
                           help='Subsample input frames x_y_z means subsample every x frame at 1st layer, '
                                'every y frame at 2nd layer etc.')
        return parser

    @staticmethod
    def attention_add_arguments(parser):
        """Add arguments for the attention."""
        group = parser.add_argument_group("E2Eschedules attention setting")
        # attention
        group.add_argument('--atype', default='dot', type=str,
                           choices=['noatt', 'dot', 'add', 'location', 'coverage',
                                    'coverage_location', 'location2d', 'location_recurrent',
                                    'multi_head_dot', 'multi_head_add', 'multi_head_loc',
                                    'multi_head_multi_res_loc'],
                           help='Type of attention architecture')
        group.add_argument('--adim', default=320, type=int,
                           help='Number of attention transformation dimensions')
        group.add_argument('--awin', default=5, type=int,
                           help='Window size for location2d attention')
        group.add_argument('--aheads', default=4, type=int,
                           help='Number of heads for multi head attention')
        group.add_argument('--aconv-chans', default=-1, type=int,
                           help='Number of attention convolution channels \
                           (negative value indicates no location-aware attention)')
        group.add_argument('--aconv-filts', default=100, type=int,
                           help='Number of attention convolution filters \
                           (negative value indicates no location-aware attention)')
        group.add_argument('--dropout-rate', default=0.0, type=float,
                           help='Dropout rate for the encoder')
        return parser

    @staticmethod
    def decoder_add_arguments(parser):
        """Add arguments for the decoder."""
        group = parser.add_argument_group("E2Eschedules encoder setting")
        group.add_argument('--dtype', default='lstm', type=str,
                           choices=['lstm', 'gru'],
                           help='Type of decoder network architecture')
        group.add_argument('--dlayers', default=1, type=int,
                           help='Number of decoder layers')
        group.add_argument('--dunits', default=320, type=int,
                           help='Number of decoder hidden units')
        group.add_argument('--dropout-rate-decoder', default=0.0, type=float,
                           help='Dropout rate for the decoder')
        group.add_argument('--sampling-probability', default=0.0, type=float,
                           help='Ratio of predicted labels fed back to decoder')
        group.add_argument('--lsm-type', const='', default='', type=str, nargs='?',
                           choices=['', 'unigram'],
                           help='Apply label smoothing with a specified distribution type')
        group.add_argument('--dec-methods', default='prior', type=str,
                            choices=[ 
                                     'mltaskbaserll_modi3',
                                     'mltaskbaserll_modi3c',
                                     'mltaskbaserllmodi3v2',
                                     'mltaskbaserllmodi3v3',
                                     'mltaskrllm3ss',
                                     ],
                            help='Choose Decoder Type (i is for forward information bpe2char while r is for backward).')
        return parser

    def __init__(self, idim, odim, args, args2=None):
        """Construct an E2Eschedules object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        super(E2Eschedules, self).__init__()
        torch.nn.Module.__init__(self)
        self.mtlalpha = args.mtlalpha
        assert 0.0 <= self.mtlalpha <= 1.0, "mtlalpha should be [0.0, 1.0]"
        print("[e2e_asrss.py] This code is for multi-task attention-based encoder-decoder architecture with ssprob=0.0.")
        print('Setting the encoder dropout %2.f' %(args.dropout_rate))
        self.epochindex = args.epochindex
        self.sampling_probability = 0.0


        self.etype = args.etype
        self.verbose = args.verbose
        # NOTE: for self.build method
        args.bpe_list = getattr(args, "bpe_list", None)
        self.bpe_list = args.bpe_list
        args.char_list = getattr(args, "char_list", None)
        self.char_list = args.char_list        
        self.outdir = args.outdir
        self.space = args.sym_space
        self.blank = args.sym_blank
        self.reporter = MltReporter()
        self.dec_methods = args.dec_methods

        # below means the last number becomes eos/sos ID
        # note that sos/eos IDs are identical
        self.sos_char = odim[0] - 1
        self.eos_char = odim[0] - 1
        self.sos_bpe = odim[1] - 1
        self.eos_bpe = odim[1] - 1

        self.dec_methods = args.dec_methods

        if (args2 == None):
            args_series = copy.copy(args)
            args_series_bpe = copy.copy(args)
        else:
            args_series, args_series_bpe = args, args2
        args_series.odim_char = odim[0] # 52
        args_series.odim_bpe = odim[1] # 496


        # subsample info
        # +1 means input (+1) and layers outputs (args.elayer)
        subsample = np.ones(args.elayers + 1, dtype=np.int)
        if args.etype.endswith("p") and not args.etype.startswith("vgg"):
            ss = args.subsample.split("_")
            for j in range(min(args.elayers + 1, len(ss))):
                subsample[j] = int(ss[j])
        else:
            logging.warning(
                'Subsampling is not performed for vgg*. It is performed in max pooling layers at CNN.')
        logging.info('subsample: ' + ' '.join([str(x) for x in subsample]))
        self.subsample = subsample

        # label smoothing info
        if args_series.lsm_type:
            logging.info("Use label smoothing with " + args_series.lsm_type)
            labeldist_char = label_smoothing_dist(args_series.odim_char, args_series.lsm_type, transcript=args_series.train_json)
            labeldist_bpe  = label_smoothing_bpe_dist(args_series.odim_bpe, args_series.lsm_type, transcript=args_series.train_json)
        else:
            labeldist_char = None
            labeldist_bpe  = None

        if getattr(args, "use_frontend", False):  # use getattr to keep compatibility
            # Relative importing because of using python3 syntax
            from espnet.nets.pytorch_backend.frontends.feature_transform \
                import feature_transform_for
            from espnet.nets.pytorch_backend.frontends.frontend \
                import frontend_for

            self.frontend = frontend_for(args, idim)
            self.feature_transform = feature_transform_for(args, (idim - 1) * 2) # TJ 2019.12.30: ?
            idim = args.n_mels
        else:
            self.frontend = None

        # encoder 
        self.enc = encoder_for(args, idim, self.subsample) # TJ 2019.12.30: encoder part: 
        
        # attention and decoder
        if args.dec_methods == 'mltaskbaserll_modi3':
            self.ctc = ctc_for(args, args_series.odim_char) # TJ 2019.12.30: CTC module         
            print('For attention : this uses the CHAR-level CTC Auxilitary Loss (mltaskbaserll_modi3)----TRY.')
            self.att_char, self.att_bpe, self.att_char2bpe = att_for(args_series), att_for(args_series_bpe), att_for(args_series_bpe)
            self.dec = decoder_base_mltaskller_modification3_for(
                            args_series, args_series.odim_char, args_series.odim_bpe,
                            self.sos_char, self.eos_char, self.sos_bpe, self.eos_bpe, self.att_char, self.att_bpe, self.att_char2bpe, labeldist_char, labeldist_bpe
                        )
        elif args.dec_methods == 'mltaskbaserll_modi3c':
            self.ctc = ctc_for(args, args_series.odim_char) # TJ 2019.12.30: CTC module         
            print('For attention : this uses the CHAR-level CTC Auxilitary Loss (mltaskbaserll_modi3c)----TRY.')
            self.att_char, self.att_bpe, self.att_char2bpe = att_for(args_series), att_for(args_series_bpe), att_for(args_series_bpe)
            self.dec = decoder_base_mltaskller_m3c_schedulesampling_for(
                            args_series, args_series.odim_char, args_series.odim_bpe,
                            self.sos_char, self.eos_char, self.sos_bpe, self.eos_bpe, self.att_char, self.att_bpe, self.att_char2bpe, labeldist_char, labeldist_bpe
                        )
        elif args.dec_methods == 'mltaskbaserllmodi3v2':
            self.ctc = ctc_for(args, args_series.odim_char) # TJ 2019.12.30: CTC module         
            print('For attention : this uses the CHAR-level CTC Auxilitary Loss (mltaskbaserllmodi3v2)----Only BPE SS operation.')
            self.att_char, self.att_bpe, self.att_char2bpe = att_for(args_series), att_for(args_series_bpe), att_for(args_series_bpe)
            self.dec = decoder_base_mltaskller_m3_schedulesampling2_for(
                            args_series, args_series.odim_char, args_series.odim_bpe,
                            self.sos_char, self.eos_char, self.sos_bpe, self.eos_bpe, self.att_char, self.att_bpe, self.att_char2bpe, labeldist_char, labeldist_bpe
                        )
        elif args.dec_methods == 'mltaskbaserllmodi3v3':
            self.ctc = ctc_for(args, args_series.odim_char) # TJ 2019.12.30: CTC module         
            print('For attention : this uses the CHAR-level CTC Auxilitary Loss (mltaskbaserllmodi3v3)----Only BPE SS operation.')
            self.att_char, self.att_bpe, self.att_char2bpe = att_for(args_series), att_for(args_series_bpe), att_for(args_series_bpe)
            self.dec = decoder_base_mltaskller_m3_schedulesampling3_for(
                            args_series, args_series.odim_char, args_series.odim_bpe,
                            self.sos_char, self.eos_char, self.sos_bpe, self.eos_bpe, self.att_char, self.att_bpe, self.att_char2bpe, labeldist_char, labeldist_bpe
                        )

        elif args.dec_methods == 'mltaskrllm3ss':
            self.ctc = ctc_for(args, args_series.odim_char) # TJ 2019.12.30: CTC module         
            print('For attention : this uses the CHAR-level CTC Auxilitary Loss (mltaskrllm3ss)----Only BPE SS operation.')
            self.att_char, self.att_bpe, self.att_char2bpe = att_for(args_series), att_for(args_series_bpe), att_for(args_series_bpe)
            self.dec = decoder_base_newmltaskr_m3ss_for(
                            args_series, args_series.odim_char, args_series.odim_bpe,
                            self.sos_char, self.eos_char, self.sos_bpe, self.eos_bpe, self.att_char, self.att_bpe, self.att_char2bpe, labeldist_char, labeldist_bpe
                        )

        else:
            print("This decoder methods is not suitable for e2e_asrss.py (%s)." %(args.dec_methods))
            exit(0)

        # weight initialization
        self.init_like_chainer() # TJ 2019.12.30: initialization

        # options for beam search
        if args.report_cer or args.report_wer:
            recog_args = {'beam_size': args.beam_size, 'penalty': args.penalty,
                          'ctc_weight': args.ctc_weight, 'maxlenratio': args.maxlenratio,
                          'minlenratio': args.minlenratio, 'lm_weight': args.lm_weight,
                          'rnnlm': args.rnnlm, 'nbest': args.nbest,
                          'space': args.sym_space, 'blank': args.sym_blank}

            self.recog_args = argparse.Namespace(**recog_args)
            self.report_cer = args.report_cer
            self.report_wer = args.report_wer
        else:
            self.report_cer = False
            self.report_wer = False
        self.rnnlm = None

        self.logzero = -10000000000.0
        self.loss = None
        self.acc = None

    def init_like_chainer(self):
        """Initialize weight like chainer.

        chainer basically uses LeCun way: W ~ Normal(0, fan_in ** -0.5), b = 0
        pytorch basically uses W, b ~ Uniform(-fan_in**-0.5, fan_in**-0.5)
        however, there are two exceptions as far as I know.
        - EmbedID.W ~ Normal(0, 1)
        - LSTM.upward.b[forget_gate_range] = 1 (but not used in NStepLSTM)
        """
        lecun_normal_init_parameters(self)
        # exceptions
        # embed weight ~ Normal(0, 1)
        self.dec.embed_char.weight.data.normal_(0, 1) ## need add more weight initialization
        self.dec.embed_bpe.weight.data.normal_(0, 1)
        # # forget-bias = 1.0
        # # https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745
        # for l in six.moves.range(len(self.dec.decoder)):
        #     set_forget_bias_to_one(self.dec.decoder[l].bias_ih)

    def forward(self, xs_pad, ilens, ys_pad):
        """E2Eschedules forward.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :return: loss value
        :rtype: torch.Tensor
        """
        [ys_pad1, ys_pad2] = ys_pad # [B,decTchar] [B,decTbpe]

        # 0. Frontend
        if self.frontend is not None:
            hs_pad, hlens, mask = self.frontend(to_torch_tensor(xs_pad), ilens)
            hs_pad, hlens = self.feature_transform(hs_pad, hlens)
        else:
            hs_pad, hlens = xs_pad, ilens

        # 1. Encoder
        hs_pad, hlens, _ = self.enc(hs_pad, hlens)

        # 2. CTC loss
        if self.mtlalpha == 0:
            loss_ctc = None
        else:
            if self.dec_methods[-1] == "b":
                loss_ctc = self.ctc(hs_pad, hlens, ys_pad2) ## this is BPE CTC loss
            else:
                loss_ctc = self.ctc(hs_pad, hlens, ys_pad1)

        # 3. attention loss
        [loss_att, aux_loss_att], [acc_char, acc_bpe] = self.dec(hs_pad, hlens, ys_pad1, ys_pad2, self.sampling_probability)
        acc1, acc2 = acc_char, acc_bpe # 
        self.acc = acc1

        # 4. compute cer without beam search
        if self.mtlalpha == 0:
            cer_ctc = None # no ctc
        else:
            cers = []
            # calculate CTC CER
            y_hats = self.ctc.argmax(hs_pad).data # [batchsize,decT]
            for i, y in enumerate(y_hats):
                y_true = ys_pad1[i]
                y_hat = y
                seq_hat = [self.bpe_list[int(idx)] for idx in y_hat if int(idx) != -1]
                seq_true = [self.bpe_list[int(idx)] for idx in y_true if int(idx) != -1]
                seq_hat_text = "".join(seq_hat).replace(self.space, ' ')
                seq_hat_text = seq_hat_text.replace(self.blank, '')
                seq_true_text = "".join(seq_true).replace(self.space, ' ')

                hyp_chars = seq_hat_text.replace(' ', '')
                ref_chars = seq_true_text.replace(' ', '')
                if len(ref_chars) > 0:
                    cers.append(editdistance.eval(hyp_chars, ref_chars) / len(ref_chars))
            cer_ctc = sum(cers) / len(cers) if cers else None

        # 5. compute cer/wer
        if self.training or not (self.report_cer or self.report_wer): # default setting
            cer, wer = 0.0, 0.0
        else:
            if self.recog_args.ctc_weight > 0.0:
                lpz = self.ctc.log_softmax(hs_pad).data
            else:
                lpz = None

            wers, cers = [], []
            nbest_hyps = self.dec.recognize_beam_batch(hs_pad, torch.tensor(hlens), lpz, self.recog_args, self.bpe_list, self.rnnlm)
            # remove <sos> and <eos>
            y_hats = [nbest_hyp[0]['yseq'][1:-1] for nbest_hyp in nbest_hyps]
            for i, y_hat in enumerate(y_hats):
                y_true = ys_pad1[i]

                seq_hat = [self.bpe_list[int(idx)] for idx in y_hat if int(idx) != -1]
                seq_true = [self.bpe_list[int(idx)] for idx in y_true if int(idx) != -1]
                seq_hat_text = "".join(seq_hat).replace(self.recog_args.space, ' ')
                seq_hat_text = seq_hat_text.replace(self.recog_args.blank, '')
                seq_true_text = "".join(seq_true).replace(self.recog_args.space, ' ')

                hyp_words = seq_hat_text.split()
                ref_words = seq_true_text.split()
                wers.append(editdistance.eval(hyp_words, ref_words) / len(ref_words))
                hyp_chars = seq_hat_text.replace(' ', '')
                ref_chars = seq_true_text.replace(' ', '')
                cers.append(editdistance.eval(hyp_chars, ref_chars) / len(ref_chars))

            wer = 0.0 if not self.report_wer else sum(wers) / len(wers)
            cer = 0.0 if not self.report_cer else sum(cers) / len(cers)

        if self.mtlalpha == 0: 
            loss_ctc_data = None
            self.loss = loss_att + aux_loss_att
            loss1, loss2 = loss_att, aux_loss_att
        else:
            loss_ctc_data = float(loss_ctc)
            self.loss = (self.mtlalpha * loss_ctc) + ((1 - self.mtlalpha) * (loss_att + aux_loss_att))
            loss1 = (self.mtlalpha * loss_ctc) + ((1 - self.mtlalpha) * (loss_att))
            loss2 = aux_loss_att

        char_loss_att_data = float(loss_att)
        bpe_loss_att_data = float(aux_loss_att)                    

        loss_data = float(self.loss)
        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            self.reporter.report(loss_ctc_data, char_loss_att_data, bpe_loss_att_data, acc1, acc2, cer_ctc, cer, wer, loss_data)
        else:
            logging.warning('loss (=%f) is not correct', loss_data)

        # Note(kamo): In order to work with DataParallel, on pytorch==0.4,
        # the return value must be torch.CudaTensor, or tuple/list/dict of it.
        # Neither CPUTensor nor float/int value can be used
        # because NCCL communicates between GPU devices.
        device = next(self.parameters()).device

        acc1 = torch.tensor([acc1], device=device) if acc1 is not None else None
        acc2 = torch.tensor([acc2], device=device) if acc2 is not None else None
        cer = torch.tensor([cer], device=device)
        wer = torch.tensor([wer], device=device)

        return self.loss, loss_ctc, char_loss_att_data+bpe_loss_att_data, acc1, acc2, cer, wer # char_loss_att_data is float()

    def scorers(self):
        """Scorers."""
        return dict(decoder=self.dec, ctc=CTCPrefixScorer(self.ctc, self.eos))

    def encode(self, x):
        """Encode acoustic features.

        :param ndarray x: input acoustic feature (T, D)
        :return: encoder outputs
        :rtype: torch.Tensor
        """
        self.eval()
        ilens = [x.shape[0]]

        # subsample frame
        x = x[::self.subsample[0], :]
        p = next(self.parameters())
        h = torch.as_tensor(x, device=p.device, dtype=p.dtype)
        # make a utt list (1) to use the same interface for encoder
        hs = h.contiguous().unsqueeze(0)

        # 0. Frontend
        if self.frontend is not None:
            enhanced, hlens, mask = self.frontend(hs, ilens)
            hs, hlens = self.feature_transform(enhanced, hlens)
        else:
            hs, hlens = hs, ilens

        # 1. encoder
        hs, _, _ = self.enc(hs, hlens)
        return hs.squeeze(0)

    def recognize(self, x, recog_args, bpe_list, rnnlm=None):
        """E2Eschedules beam search.

        :param ndarray x: input acoustic feature (T, D)
        :param Namespace recog_args: argument Namespace containing options
        :param list bpe_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        hs = self.encode(x).unsqueeze(0)
        # calculate log P(z_t|X) for CTC scores
        if recog_args.ctc_weight > 0.0:
            lpz = self.ctc.log_softmax(hs)[0]
        else:
            lpz = None

        # 2. Decoder
        # decode the first utterance
        y = self.dec.recognize_beam(hs[0], lpz, recog_args, bpe_list, rnnlm)
        return y

    def recognize_batch(self, xs, recog_args, bpe_list, rnnlm=None):
        """E2Eschedules beam search.

        :param list xs: list of input acoustic feature arrays [(T_1, D), (T_2, D), ...]
        :param Namespace recog_args: argument Namespace containing options
        :param list bpe_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        prev = self.training
        self.eval()
        ilens = np.fromiter((xx.shape[0] for xx in xs), dtype=np.int64)

        # subsample frame
        xs = [xx[::self.subsample[0], :] for xx in xs]
        xs = [to_device(self, to_torch_tensor(xx).float()) for xx in xs]
        xs_pad = pad_list(xs, 0.0)

        # 0. Frontend
        if self.frontend is not None:
            enhanced, hlens, mask = self.frontend(xs_pad, ilens)
            hs_pad, hlens = self.feature_transform(enhanced, hlens)
        else:
            hs_pad, hlens = xs_pad, ilens

        # 1. Encoder
        hs_pad, hlens, _ = self.enc(hs_pad, hlens)

        # calculate log P(z_t|X) for CTC scores
        if recog_args.ctc_weight > 0.0:
            lpz = self.ctc.log_softmax(hs_pad)
            normalize_score = False
        else:
            lpz = None
            normalize_score = True

        # 2. Decoder
        hlens = torch.tensor(list(map(int, hlens)))  # make sure hlens is tensor
        y = self.dec.recognize_beam_batch(hs_pad, hlens, lpz, recog_args, bpe_list,
                                          rnnlm, normalize_score=normalize_score)

        if prev:
            self.train()
        return y

    def enhance(self, xs):
        """Forward only in the frontend stage.

        :param ndarray xs: input acoustic feature (T, C, F)
        :return: enhaned feature
        :rtype: torch.Tensor
        """
        if self.frontend is None:
            raise RuntimeError('Frontend does\'t exist')
        prev = self.training
        self.eval()
        ilens = np.fromiter((xx.shape[0] for xx in xs), dtype=np.int64)

        # subsample frame
        xs = [xx[::self.subsample[0], :] for xx in xs]
        xs = [to_device(self, to_torch_tensor(xx).float()) for xx in xs]
        xs_pad = pad_list(xs, 0.0)
        enhanced, hlensm, mask = self.frontend(xs_pad, ilens)
        if prev:
            self.train()
        return enhanced.cpu().numpy(), mask.cpu().numpy(), ilens

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad):
        """E2Eschedules attention calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :return: attention weights with the following shape,
            1) multi-head case => attention weights (B, H, Lmax, Tmax),
            2) other case => attention weights (B, Lmax, Tmax).
        :rtype: float ndarray
        """
        with torch.no_grad():
            [ys_pad1, ys_pad2] = ys_pad   
            # 0. Frontend
            if self.frontend is not None:
                hs_pad, hlens, mask = self.frontend(to_torch_tensor(xs_pad), ilens)
                hs_pad, hlens = self.feature_transform(hs_pad, hlens)
            else:
                hs_pad, hlens = xs_pad, ilens

            # 1. Encoder
            hpad, hlens, _ = self.enc(hs_pad, hlens)

            # 2. Decoder
            att_ws = self.dec.calculate_all_attentions(hpad, hlens, ys_pad1, ys_pad2, self.sampling_probability)

        return att_ws

    def subsample_frames(self, x):
        """Subsample speeh frames in the encoder."""
        # subsample frame
        x = x[::self.subsample[0], :]
        ilen = [x.shape[0]]
        h = to_device(self, torch.from_numpy(
            np.array(x, dtype=np.float32)))
        h.contiguous()
        return h, ilen
