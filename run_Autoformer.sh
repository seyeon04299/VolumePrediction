#!/bin/bash

# ------------------ Set hyperparameter tuning permutation options-----------------------------------
MODEL_NAME=("Transformer" "Autoformer")
OPTIMIZER_G=("0.0001" "0.00005" "0.001" "0.005")
D_MODEL=("512" "256" "128" "1024")
E_LAYERS=("2" "1" "3")
D_LAYERS=("1" "2")
N_HEADS=("8" "16")
D_FF=("2048" "1024")
SEQ_LEN=("48")
LABEL_LEN=("32")
PRED_LEN=("24")
SEC=("5")

for MODEL in ${MODEL_NAME[@]}
do
    for OPT_G in ${OPTIMIZER_G[@]}
    do
        for D_MOD in ${D_MODEL[@]}
        do
            for E_LAYER in ${E_LAYERS[@]}
            do
                for DFF in ${D_FF[@]}
                do
                    for D_LAYER in ${D_LAYERS[@]}
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
                                            LOGGINGNAME=_"$MODEL"+"sec""$sec"+"ALLDATA"+"seq"_"$seqlen"+"lab"_"$lablen"+"pred"_"$predlen"+"lrG"_"$OPT_G"+"dmodel"_"$D_MOD"+"Elayers"_"$E_LAYER"+"nheads"_"$N_HEAD"+"Dlayers"_"$D_LAYER"+"dff"_"$DFF"+"nopeak"+"dropout0.1"
                                            jq --arg a "$MODEL" '.name = $a' ./config/config_Autoformer.json | sponge ./config/config_Autoformer.json
                                            jq --arg a "$MODEL" '.arch_G.type = $a' ./config/config_Autoformer.json | sponge ./config/config_Autoformer.json
                                            jq --arg a "$MODEL" '.trainer.model_name = $a' ./config/config_Autoformer.json | sponge ./config/config_Autoformer.json
                                            jq --arg a "$LOGGINGNAME" '.run_name = $a' ./config/config_Autoformer.json | sponge ./config/config_Autoformer.json
                                            jq --arg a "$OPT_G" '.optimizer_G.args.lr = $a' ./config/config_Autoformer.json | sponge ./config/config_Autoformer.json
                                            jq --arg a "$D_MOD" '.arch_G.args.d_model = $a' ./config/config_Autoformer.json | sponge ./config/config_Autoformer.json
                                            jq --arg a "$DFF" '.arch_G.args.d_ff = $a' ./config/config_Autoformer.json | sponge ./config/config_Autoformer.json
                                            jq --arg a "$N_HEAD" '.arch_G.args.n_heads = $a' ./config/config_Autoformer.json | sponge ./config/config_Autoformer.json
                                            jq --arg a "$seqlen" '.data_loader.args.seq_len = $a' ./config/config_Autoformer.json | sponge ./config/config_Autoformer.json
                                            jq --arg a "$predlen" '.data_loader.args.pred_len = $a' ./config/config_Autoformer.json | sponge ./config/config_Autoformer.json
                                            jq --arg a "$lablen" '.data_loader.args.label_len = $a' ./config/config_Autoformer.json | sponge ./config/config_Autoformer.json
                                            jq --arg a "$sec" '.data_loader.args.sec = $a' ./config/config_Autoformer.json | sponge ./config/config_Autoformer.json
                                            
                                            echo $LOGGINGNAME
                                            python3 ./train_Autoformer.py --device 0,1\
                                            -c ./config/config_Autoformer.json\
                                            --MODEL "$MODEL"\
                                            --OPT_G "$OPT_G"\
                                            --D_MOD "$D_MOD"\
                                            --E_LAYERS "$E_LAYER"\
                                            --DFF "$DFF"\
                                            --D_LAYERS "$D_LAYER"\
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
    done
done
