#!/bin/bash

# ------------------ Set hyperparameter tuning permutation options-----------------------------------
OPTIMIZER_D=("0.0001")
OPTIMIZER_G=("0.0001")
D_MODEL=("512")
EMBEDDING_DIM=("120")
D_FF=("2048")
SEQ_LEN=("45")
LMBDA=("1")

# "200")

for OPT_D in ${OPTIMIZER_D[@]}
do
    for OPT_G in ${OPTIMIZER_G[@]}
    do
        for D_MOD in ${D_MODEL[@]}
        do
            for EMB_DIM in ${EMBEDDING_DIM[@]}
            do
                for DFF in ${D_FF[@]}
                do
                    for seqlen in ${SEQ_LEN[@]}
                    do
                        for lmbda in ${LMBDA[@]}
                        do
                            NAME="ASTransformer"
                            LOGGINGNAME=_"ASTransformer"+"lr_D"_"$OPT_D"+"lr_G"_"$OPT_G"+"d_model"_"$D_MOD"+"embed_dim"_"$EMB_DIM"+"d_ff"_"$DFF"+"seq_len"_"$seqlen"+"lmbda"_"$lmbda"
                            jq --arg a "$NAME" '.name = $a' ./config/config_AST.json | sponge ./config/config_AST.json
                            jq --arg a "$LOGGINGNAME" '.run_name = $a' ./config/config_AST.json | sponge ./config/config_AST.json
                            
                            echo $LOGGINGNAME
                            python3 ./train_AST.py --device 0\
                            -c ./config/config_AST.json\
                            --OPT_D "$OPT_D"\
                            --OPT_G "$OPT_G"\
                            --D_MOD "$D_MOD"\
                            --EMB_DIM "$EMB_DIM"\
                            --DFF "$DFF"\
                            --LMBDA "$lmbda"
                        done
                    done
                done
            done
        done
    done
done
