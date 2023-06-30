#!/bin/bash

# ------------------ Set hyperparameter tuning permutation options-----------------------------------
OPTIMIZER_e=("0.001")
OPTIMIZER_r=("0.001")
OPTIMIZER_s=("0.001")
OPTIMIZER_g=("0.001")
OPTIMIZER_d=("0.001")
DIS_THRES=("0.15")
GAMMA=("1")

# "200")

for OPT_E in ${OPTIMIZER_e[@]}
do
    for OPT_R in ${OPTIMIZER_r[@]}
    do
        for OPT_S in ${OPTIMIZER_s[@]}
        do
            for OPT_G in ${OPTIMIZER_g[@]}
            do
                for OPT_D in ${OPTIMIZER_d[@]}
                do
                    for dis_thres in ${DIS_THRES[@]}
                    do
                        for gamma in ${GAMMA[@]}
                        do
                            NAME="TimeGAN"
                            LOGGINGNAME=_"TimeGAN"+"lr"_"$OPT_E"
                            jq --arg a "$NAME" '.name = $a' ./config/config_TimeGAN.json | sponge ./config/config_TimeGAN.json
                            jq --arg a "$LOGGINGNAME" '.run_name = $a' ./config/config_TimeGAN.json | sponge ./config/config_TimeGAN.json
                            
                            echo $LOGGINGNAME
                            python3 ./train_TimeGAN.py --device 0\
                            -c ./config/config_TimeGAN.json\
                            --opt_E "$OPT_G"\
                            --opt_R "$OPT_R"\
                            --opt_G "$OPT_G"\
                            --opt_D "$OPT_D"\
                            --opt_S "$OPT_S"\
                            --dis_thres "$dis_thres"\
                            --gamma "$gamma"
                        done
                    done
                done
            done
        done
    done
done
