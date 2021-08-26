#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# ./run_tuning.sh --stage 3 --tag baseline_prior_seed20000_tuningrnn --seed 20000 -ngpu 1 --igpu 2

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=0        # start from 0 if you need to start from data preparation
stop_stage=100
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
igpu=0
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false

preprocess_config=conf/specaug.yaml 
train_config=conf/train.yaml
decode_config=conf/decode.yaml

# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
n_average=1

# data
swbd1_dir=/home/jtang/data/SWBD/LDC97S62
eval2000_dir="/home/jtang/data/SWBD/LDC2002S09/hub5e_00 /home/jtang/data/SWBD/LDC2002T43"
rt03_dir=/home/jtang/data/SWBD/LDC2007S10

# network architecture
# encoder related
etype=vggblstmp     # encoder architecture type
elayers=4
eunits=1000
eprojs=1000
subsample=1_2_2_1_1 # skip every n frame from input to nth layers
# decoder related
dlayers=1
dunits=1000
# attention related
atype=location
adim=1000
awin=5
aheads=4
aconv_chans=10
aconv_filts=100

# bpemode (unigram or bpe)
nbpe=2000
bpemode=bpe

# exp tag
tag="" # tag for managing experiments.
dec_methods="prior"
multilabel=1
epochs=21
seed=1
model_module="Stand"
batchsize=30
dec_level="BPE"
adjust_closectc_weight=0.0
closethreshold=0.50
decodingdir="bpe1char"
dec_njs=3
mtlbeta=0.00
closebeta_vals=0.80
testset="eval2000"
maindir="AUXLargerMLN"

# label smoothing
lsm_type=unigram
lsm_weight=0.05

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_nodup
train_dev=train_dev
recog_set="train_dev eval2000 rt03"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    # local/swbd1_data_download.sh ${swbd1_dir}
    local/swbd1_data_unzip.sh ${swbd1_dir}
    local/swbd1_prepare_dict.sh
    local/swbd1_data_prep.sh ${swbd1_dir}
    local/eval2000_data_prep.sh ${eval2000_dir}
    local/rt03_data_prep.sh ${rt03_dir}
    # upsample audio from 8k to 16k to make a recipe consistent with others
    for x in train eval2000 rt03; do
        sed -i.bak -e "s/$/ sox -R -t wav - -t wav - rate 16000 dither | /" data/${x}/wav.scp
    done
    # normalize eval2000 and rt03 texts by
    # 1) convert upper to lower
    # 2) remove tags (%AH) (%HESITATION) (%UH)
    # 3) remove <B_ASIDE> <E_ASIDE>
    # 4) remove "(" or ")"
    for x in eval2000 rt03; do
        cp data/${x}/text data/${x}/text.org
        paste -d "" \
            <(cut -f 1 -d" " data/${x}/text.org) \
            <(awk '{$1=""; print tolower($0)}' data/${x}/text.org | perl -pe 's| \(\%.*\)||g' | perl -pe 's| \<.*\>||g' | sed -e "s/(//g" -e "s/)//g") \
            | sed -e 's/\s\+/ /g' > data/${x}/text
        # rm data/${x}/text.org
    done
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for x in train eval2000 rt03; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 32 --write_utt2num_frames true \
            data/${x} exp/make_fbank/${x} ${fbankdir}
        utils/fix_data_dir.sh data/${x}
    done

    utils/subset_data_dir.sh --first data/train 4000 data/${train_dev} # 5hr 6min
    n=$(($(wc -l < data/train/segments) - 4000))
    utils/subset_data_dir.sh --last data/train ${n} data/train_nodev
    utils/data/remove_dup_utts.sh 300 data/train_nodev data/${train_set} # 286hr

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features for training
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{10,11,12,13}/${USER}/espnet-data/egs/swbd/asr1/dump/${train_set}/delta${do_delta}/storage \
        ${feat_tr_dir}/storage
    fi
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{10,11,12,13}/${USER}/espnet-data/egs/swbd/asr1/dump/${train_dev}/delta${do_delta}/storage \
        ${feat_dt_dir}/storage
    fi
    dump.sh --cmd "$train_cmd" --nj 32 --do_delta ${do_delta} \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 10 --do_delta ${do_delta} \
        data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 10 --do_delta ${do_delta} \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
fi

dict=data/lang_char/${train_set}_${bpemode}${nbpe}_units.txt
bpemodel=data/lang_char/${train_set}_${bpemode}${nbpe}

echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_char/
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC

    # map acronym such as p._h._d. to p h d for train_set& dev_set
    cp data/${train_set}/text data/${train_set}/text.backup
    cp data/${train_dev}/text data/${train_dev}/text.backup
    sed -i 's/\._/ /g; s/\.//g; s/them_1/them/g' data/${train_set}/text
    sed -i 's/\._/ /g; s/\.//g; s/them_1/them/g' data/${train_dev}/text

    echo "make a dictionary"
    cut -f 2- -d" " data/${train_set}/text > data/lang_char/input.txt

    # Please make sure sentencepiece is installed
    spm_train --input=data/lang_char/input.txt \
            --model_prefix=${bpemodel} \
            --vocab_size=${nbpe} \
            --character_coverage=1.0 \
            --model_type=${bpemode} \
            --model_prefix=${bpemodel} \
            --input_sentence_size=100000000 \
            --bos_id=-1 \
            --eos_id=-1 \
            --unk_id=0 \
            --user_defined_symbols="[laughter],[noise],[vocalized-noise]"

    spm_encode --model=${bpemodel}.model --output_format=piece < data/lang_char/input.txt | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    echo "make json files"
    data2json.sh --feat ${feat_tr_dir}/feats.scp --bpecode ${bpemodel}.model \
        data/${train_set} ${dict} > ${feat_tr_dir}/data_${bpemode}${nbpe}.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp --bpecode ${bpemodel}.model \
        data/${train_dev} ${dict} > ${feat_dt_dir}/data_${bpemode}${nbpe}.json

    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --feat ${feat_recog_dir}/feats.scp --allow-one-column true \
            --bpecode ${bpemodel}.model data/${rtask} ${dict} > ${feat_recog_dir}/data_${bpemode}${nbpe}.json
    done
fi

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    if ${do_delta}; then
    expname=${expname}_delta
    fi
    if [ -n "${preprocess_config}" ]; then
    expname=${expname}_$(basename ${preprocess_config%.*})
    fi
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=base40exps/${maindir}/${expname}
mkdir -p ${expdir}

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Network Training(rnn)."
    train_config="conf/tuning/train_rnn.yaml"
    
    if [[ "${multilabel}" -eq 1 ]]; then
        echo "Data Usage : This is only a BPE data."
        train_json=${feat_tr_dir}/data_${bpemode}${nbpe}.json
        valid_json=${feat_dt_dir}/data_${bpemode}${nbpe}.json
    elif [[ "${multilabel}" -eq 2 ]]; then
        echo "Data Usage : This is only a BPE&CHAR data."
        train_json=${feat_tr_dir}/data_joint.json
        valid_json=${feat_dt_dir}/data_joint.json            
    fi

    export CUDA_VISIBLE_DEVICES=$igpu
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --etype ${etype} \
        --elayers ${elayers} \
        --eunits ${eunits} \
        --eprojs ${eprojs} \
        --subsample ${subsample} \
        --dlayers ${dlayers} \
        --dunits ${dunits} \
        --atype ${atype} \
        --adim ${adim} \
        --awin ${awin} \
        --aconv-chans ${aconv_chans} \
        --aconv-filts ${aconv_filts} \
        --lsm-type ${lsm_type} \
        --lsm-weight ${lsm_weight} \
        --adjust-closectc-weight ${adjust_closectc_weight} \
        --closethreshold $closethreshold \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --chardict data/lang_1char/${train_set}_units.txt \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --seed ${seed} \
        --model_module ${model_module} \
        --epochs ${epochs} \
        --batch-size ${batchsize} \
        --dec-methods ${dec_methods} \
        --multi-label ${multilabel} \
        --mtlbeta ${mtlbeta} \
        --closebeta_vals ${closebeta_vals} \
        --train-json ${train_json} \
        --valid-json ${valid_json}  &
    echo "stage 3: This is only send training commands."
    exit 1;
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Decoding"
    decode_config="conf/tuning/decode_rnn_noctc.yaml"
    export CUDA_VISIBLE_DEVICES=$igpu
    nj=${dec_njs}
    recog_set=${testset}
    pids=() # initialize pids
    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${decodingdir}_${rtask}_$(basename ${decode_config%.*})
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json

        #### use CPU for decoding
        ngpu=1

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --dec-level ${dec_level} \
            --CHOICE_SCORE_TYPES ${decodingdir} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/model.acc.best

    # this is required for local/score_sclite.sh to get hyp.wrd.trn
        score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}
    if [[ "${decode_dir}" =~ "eval2000" ]]; then
            local/score_sclite.sh data/eval2000 ${expdir}/${decode_dir}
    elif [[ "${decode_dir}" =~ "rt03" ]]; then
        local/score_sclite.sh data/rt03 ${expdir}/${decode_dir}
    fi
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
    exit 1;
fi

decode_config="conf/tuning/decode_rnn_noctc.yaml"
export CUDA_VISIBLE_DEVICES=$igpu
nj=${dec_njs}
ngpu=1

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    ###########################################################################
    pids=() # initialize pids
    rtask="eval2000"
    decodingdir="bpeonly"
    decode_dir=decode_${decodingdir}_${rtask}_$(basename ${decode_config%.*})
    feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
    # split data
    splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json
    #### use CPU for decoding
    ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
        asr_recog.py \
        --config ${decode_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --dec-level ${dec_level} \
        --CHOICE_SCORE_TYPES ${decodingdir} \
        --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
        --result-label ${expdir}/${decode_dir}/data.JOB.json \
        --model ${expdir}/results/model.acc.best
    # this is required for local/score_sclite.sh to get hyp.wrd.trn
        score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}
    if [[ "${decode_dir}" =~ "eval2000" ]]; then
            local/score_sclite.sh data/eval2000 ${expdir}/${decode_dir}
    elif [[ "${decode_dir}" =~ "rt03" ]]; then
        local/score_sclite.sh data/rt03 ${expdir}/${decode_dir}
    fi
    ###########################################################################
    wait;
    pids=() # initialize pids
    rtask="eval2000"
    decodingdir="bpe2char"
    decode_dir=decode_${decodingdir}_${rtask}_$(basename ${decode_config%.*})
    feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
    # split data
    splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json
    #### use CPU for decoding
    ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
        asr_recog.py \
        --config ${decode_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --dec-level ${dec_level} \
        --CHOICE_SCORE_TYPES ${decodingdir} \
        --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
        --result-label ${expdir}/${decode_dir}/data.JOB.json \
        --model ${expdir}/results/model.acc.best
    # this is required for local/score_sclite.sh to get hyp.wrd.trn
        score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}
    if [[ "${decode_dir}" =~ "eval2000" ]]; then
            local/score_sclite.sh data/eval2000 ${expdir}/${decode_dir}
    elif [[ "${decode_dir}" =~ "rt03" ]]; then
        local/score_sclite.sh data/rt03 ${expdir}/${decode_dir}
    fi
    ###########################################################################        
    wait;
    pids=() # initialize pids
    rtask="train_dev"
    decodingdir="bpe2char"
    decode_dir=decode_${decodingdir}_${rtask}_$(basename ${decode_config%.*})
    feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
    # split data
    splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json
    #### use CPU for decoding
    ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
        asr_recog.py \
        --config ${decode_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --dec-level ${dec_level} \
        --CHOICE_SCORE_TYPES ${decodingdir} \
        --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
        --result-label ${expdir}/${decode_dir}/data.JOB.json \
        --model ${expdir}/results/model.acc.best
    # this is required for local/score_sclite.sh to get hyp.wrd.trn
        score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}
    if [[ "${decode_dir}" =~ "eval2000" ]]; then
            local/score_sclite.sh data/eval2000 ${expdir}/${decode_dir}
    elif [[ "${decode_dir}" =~ "rt03" ]]; then
        local/score_sclite.sh data/rt03 ${expdir}/${decode_dir}
    fi
    ###########################################################################
    wait;
    pids=() # initialize pids
    rtask="train_dev"
    decodingdir="bpeonly"
    decode_dir=decode_${decodingdir}_${rtask}_$(basename ${decode_config%.*})
    feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
    # split data
    splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json
    #### use CPU for decoding
    ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
        asr_recog.py \
        --config ${decode_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --dec-level ${dec_level} \
        --CHOICE_SCORE_TYPES ${decodingdir} \
        --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
        --result-label ${expdir}/${decode_dir}/data.JOB.json \
        --model ${expdir}/results/model.acc.best
    # this is required for local/score_sclite.sh to get hyp.wrd.trn
        score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}
    if [[ "${decode_dir}" =~ "eval2000" ]]; then
            local/score_sclite.sh data/eval2000 ${expdir}/${decode_dir}
    elif [[ "${decode_dir}" =~ "rt03" ]]; then
        local/score_sclite.sh data/rt03 ${expdir}/${decode_dir}
    fi
    ###########################################################################
    wait;
    pids=() # initialize pids
    rtask="rt03"
    decodingdir="bpe2char"
    decode_dir=decode_${decodingdir}_${rtask}_$(basename ${decode_config%.*})
    feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
    # split data
    splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json
    #### use CPU for decoding
    ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
        asr_recog.py \
        --config ${decode_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --dec-level ${dec_level} \
        --CHOICE_SCORE_TYPES ${decodingdir} \
        --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
        --result-label ${expdir}/${decode_dir}/data.JOB.json \
        --model ${expdir}/results/model.acc.best
    # this is required for local/score_sclite.sh to get hyp.wrd.trn
        score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}
    if [[ "${decode_dir}" =~ "eval2000" ]]; then
            local/score_sclite.sh data/eval2000 ${expdir}/${decode_dir}
    elif [[ "${decode_dir}" =~ "rt03" ]]; then
        local/score_sclite.sh data/rt03 ${expdir}/${decode_dir}
    fi
    ###########################################################################
    wait;
    pids=() # initialize pids
    rtask="rt03"
    decodingdir="bpeonly"
    decode_dir=decode_${decodingdir}_${rtask}_$(basename ${decode_config%.*})
    feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
    # split data
    splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json
    #### use CPU for decoding
    ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
        asr_recog.py \
        --config ${decode_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --dec-level ${dec_level} \
        --CHOICE_SCORE_TYPES ${decodingdir} \
        --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
        --result-label ${expdir}/${decode_dir}/data.JOB.json \
        --model ${expdir}/results/model.acc.best
    # this is required for local/score_sclite.sh to get hyp.wrd.trn
        score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}
    if [[ "${decode_dir}" =~ "eval2000" ]]; then
            local/score_sclite.sh data/eval2000 ${expdir}/${decode_dir}
    elif [[ "${decode_dir}" =~ "rt03" ]]; then
        local/score_sclite.sh data/rt03 ${expdir}/${decode_dir}
    fi
fi
