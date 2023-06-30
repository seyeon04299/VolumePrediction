#!/bin/bash

# ------------------ Set hyperparameter tuning permutation options-----------------------------------
MODEL_NAME=("ETSFormer")
OPTIMIZER_G=("0.0001")
D_MODEL=("512")
E_LAYERS=("2")
N_HEADS=("8")
SEQ_LEN=("30")
LABEL_LEN=("30")
PRED_LEN=("30")
SEC=("60")

for MODEL in ${MODEL_NAME[@]}
do
    for OPT_G in ${OPTIMIZER_G[@]}
    do
        for D_MOD in ${D_MODEL[@]}
        do
            for E_LAYER in ${E_LAYERS[@]}
            do
                for N_HEAD in ${N_HEADS[@]}
                do
                    for seqlen in ${SEQ_LEN[@]}
                    do
                        for lablen in ${LABEL_LEN[@]}
                        do
                            for predlen in ${PRED_LEN[@]}
                            do
                                for sec in ${SEC[@]}
                                do
                                    LOGGINGNAME=_"$MODEL"+"sec""$sec"+"seq"_"$seqlen"+"lab"_"$lablen"+"pred"_"$predlen"+"lrG"_"$OPT_G"+"dmodel"_"$D_MOD"+"Elayers"_"$E_LAYER"+"nheads"_"$N_HEAD"+"Dlayers"_"$D_LAYER"+"dff"_"$DFF"+"nopeak"+"dropout0.2"
                                    jq --arg a "$MODEL" '.name = $a' ./config/config_ETSFormer.json | sponge ./config/config_ETSFormer.json
                                    jq --arg a "$MODEL" '.arch_G.type = $a' ./config/config_ETSFormer.json | sponge ./config/config_ETSFormer.json
                                    jq --arg a "$LOGGINGNAME" '.run_name = $a' ./config/config_ETSFormer.json | sponge ./config/config_ETSFormer.json
                                    jq --arg a "$OPT_G" '.optimizer_G.args.lr = $a' ./config/config_ETSFormer.json | sponge ./config/config_ETSFormer.json
                                    jq --arg a "$D_MOD" '.arch_G.args.d_model = $a' ./config/config_ETSFormer.json | sponge ./config/config_ETSFormer.json
                                    jq --arg a "$N_HEAD" '.arch_G.args.n_heads = $a' ./config/config_ETSFormer.json | sponge ./config/config_ETSFormer.json
                                    jq --arg a "$seqlen" '.data_loader.args.seq_len = $a' ./config/config_ETSFormer.json | sponge ./config/config_ETSFormer.json
                                    jq --arg a "$predlen" '.data_loader.args.pred_len = $a' ./config/config_ETSFormer.json | sponge ./config/config_ETSFormer.json
                                    jq --arg a "$lablen" '.data_loader.args.label_len = $a' ./config/config_ETSFormer.json | sponge ./config/config_ETSFormer.json
                                    jq --arg a "$sec" '.data_loader.args.sec = $a' ./config/config_ETSFormer.json | sponge ./config/config_ETSFormer.json
                                    
                                    echo $LOGGINGNAME
                                    python3 ./train_ETSFormer.py --device 0,1\
                                    -c ./config/config_ETSFormer.json\
                                    --MODEL "$MODEL"\
                                    --OPT_G "$OPT_G"\
                                    --D_MOD "$D_MOD"\
                                    --E_LAYERS "$E_LAYER"\
                                    --N_HEADS "$N_HEAD"\
                                    --PRED_LEN "$predlen"\
                                    --LABEL_LEN "$lablen"\
                                    --SEQ_LEN "$seqlen"\
                                    --SEC "$sec"
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
