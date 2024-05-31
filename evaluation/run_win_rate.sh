#!/bin/bash

models=("sftExp53_llama2-70b_stage3" "sftExp53_llama2-70b_stage2" "sftExp53_llama2-70b_stage1")
for model_compared in ${models[*]}
do
    for eval_file in 'anthropic' 'oasst' 'koala' 'sinstruct' 'wizardlm' 'vicuna'
    do
        k1=sftExp35_llama2-70b_baseline
        k2=$model_compared

        python win_tie_loss_stat.py \
            -i1 ${k1}-${k2}-${eval_file}.json \
            -k1 $k1 \
            -i2 ${k2}-${k1}-${eval_file}.json \
            -k2 $k2 \
            --output_dir ./ \
            --dst $eval_file
    done
done

