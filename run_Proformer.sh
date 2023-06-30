#!/bin/bash

# ------------------ Set hyperparameter tuning permutation options-----------------------------------
MODEL_NAME=("TransProformer")
OPTIMIZER_G0=("0.0001" "0.00005")
OPTIMIZER_G1=("0.0001" "0.001")
OPTIMIZER_G2=("0.0001" "0.001")
OPTIMIZER_G3=("0.001")
OPTIMIZER_G4=("0.001")
OPTIMIZER_G5=("0.001")
OPTIMIZER_G6=("0.0001")
D_MODEL=("256" "512")
E_LAYERS=("2")
D_LAYERS=("1")
N_HEADS=("8")
D_FF=("2048")
SEQ_LEN=("48")
LABEL_LEN=("48")
PRED_LEN=("4632")
SEC=("5")

for MODEL in ${MODEL_NAME[@]}
do
    for OPT_G0 in ${OPTIMIZER_G0[@]}
    do
        for OPT_G1 in ${OPTIMIZER_G1[@]}
        do
            for OPT_G2 in ${OPTIMIZER_G2[@]}
            do
                for OPT_G3 in ${OPTIMIZER_G3[@]}
                do
                    for OPT_G4 in ${OPTIMIZER_G4[@]}
                    do
                        for OPT_G5 in ${OPTIMIZER_G5[@]}
                        do
                            for OPT_G6 in ${OPTIMIZER_G6[@]}
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
                                                                    # LOGGINGNAME=_"$MODEL"+"sec""$sec"+"seq"_"$seqlen"+"lab"_"$lablen"+"pred"_"$predlen"+"lrG"_"$OPT_G"+"dmodel"_"$D_MOD"+"Elayers"_"$E_LAYER"+"nheads"_"$N_HEAD"+"Dlayers"_"$D_LAYER"+"dff"_"$DFF"+"nopeak"+"dropout0.1"
                                                                    LOGGINGNAME=_"$MODEL"+"gamma"_"1"+"opt5_opt6_0.001"
                                                                    jq --arg a "$MODEL" '.name = $a' ./config/config_Proformer.json | sponge ./config/config_Proformer.json
                                                                    jq --arg a "$MODEL" '.arch_G0.type = $a' ./config/config_Proformer.json | sponge ./config/config_Proformer.json
                                                                    jq --arg a "$MODEL" '.trainer.model_name = $a' ./config/config_Proformer.json | sponge ./config/config_Proformer.json
                                                                    jq --arg a "$LOGGINGNAME" '.run_name = $a' ./config/config_Proformer.json | sponge ./config/config_Proformer.json
                                                                    jq --arg a "$OPT_G0" '.optimizer_G0.args.lr = $a' ./config/config_Proformer.json | sponge ./config/config_Proformer.json
                                                                    jq --arg a "$OPT_G1" '.optimizer_G1.args.lr = $a' ./config/config_Proformer.json | sponge ./config/config_Proformer.json
                                                                    jq --arg a "$OPT_G2" '.optimizer_G2.args.lr = $a' ./config/config_Proformer.json | sponge ./config/config_Proformer.json
                                                                    jq --arg a "$OPT_G3" '.optimizer_G3.args.lr = $a' ./config/config_Proformer.json | sponge ./config/config_Proformer.json
                                                                    jq --arg a "$OPT_G4" '.optimizer_G4.args.lr = $a' ./config/config_Proformer.json | sponge ./config/config_Proformer.json
                                                                    jq --arg a "$OPT_G5" '.optimizer_G5.args.lr = $a' ./config/config_Proformer.json | sponge ./config/config_Proformer.json
                                                                    jq --arg a "$OPT_G6" '.optimizer_G6.args.lr = $a' ./config/config_Proformer.json | sponge ./config/config_Proformer.json
                                                                    jq --arg a "$D_MOD" '.arch_G0.args.d_model = $a' ./config/config_Proformer.json | sponge ./config/config_Proformer.json
                                                                    jq --arg a "$D_MOD" '.arch_G1.args.d_model = $a' ./config/config_Proformer.json | sponge ./config/config_Proformer.json
                                                                    jq --arg a "$D_MOD" '.arch_G2.args.d_model = $a' ./config/config_Proformer.json | sponge ./config/config_Proformer.json
                                                                    jq --arg a "$D_MOD" '.arch_G3.args.d_model = $a' ./config/config_Proformer.json | sponge ./config/config_Proformer.json
                                                                    jq --arg a "$D_MOD" '.arch_G4.args.d_model = $a' ./config/config_Proformer.json | sponge ./config/config_Proformer.json
                                                                    jq --arg a "$D_MOD" '.arch_G5.args.d_model = $a' ./config/config_Proformer.json | sponge ./config/config_Proformer.json
                                                                    jq --arg a "$D_MOD" '.arch_G6.args.d_model = $a' ./config/config_Proformer.json | sponge ./config/config_Proformer.json
                                                                    jq --arg a "$DFF" '.arch_G0.args.d_ff = $a' ./config/config_Proformer.json | sponge ./config/config_Proformer.json
                                                                    jq --arg a "$N_HEAD" '.arch_G0.args.n_heads = $a' ./config/config_Proformer.json | sponge ./config/config_Proformer.json
                                                                    jq --arg a "$seqlen" '.data_loader.args.seq_len = $a' ./config/config_Proformer.json | sponge ./config/config_Proformer.json
                                                                    jq --arg a "$predlen" '.data_loader.args.pred_len = $a' ./config/config_Proformer.json | sponge ./config/config_Proformer.json
                                                                    jq --arg a "$lablen" '.data_loader.args.label_len = $a' ./config/config_Proformer.json | sponge ./config/config_Proformer.json
                                                                    jq --arg a "$sec" '.data_loader.args.sec = $a' ./config/config_Proformer.json | sponge ./config/config_Proformer.json
                                                                    
                                                                    echo $LOGGINGNAME
                                                                    python3 ./train_Proformer.py --device 0,1\
                                                                    -c ./config/config_Proformer.json\
                                                                    --MODEL "$MODEL"\
                                                                    --OPT_G0 "$OPT_G0"\
                                                                    --OPT_G1 "$OPT_G1"\
                                                                    --OPT_G2 "$OPT_G2"\
                                                                    --OPT_G3 "$OPT_G3"\
                                                                    --OPT_G4 "$OPT_G4"\
                                                                    --OPT_G5 "$OPT_G5"\
                                                                    --OPT_G6 "$OPT_G6"\
                                                                    --D_MOD0 "$D_MOD"\
                                                                    --D_MOD1 "$D_MOD"\
                                                                    --D_MOD2 "$D_MOD"\
                                                                    --D_MOD3 "$D_MOD"\
                                                                    --D_MOD4 "$D_MOD"\
                                                                    --D_MOD5 "$D_MOD"\
                                                                    --D_MOD6 "$D_MOD"\
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
                    done
                done
            done
        done
    done
done
