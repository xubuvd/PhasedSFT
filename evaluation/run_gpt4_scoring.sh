#!/bin/bash
# gpt-3.5-turbo-0613
# gpt-4-0613

models=("sftExp53_llama2-70b_stage3" "sftExp53_llama2-70b_stage2" "sftExp53_llama2-70b_stage1")
for model in ${models[*]}
do
    for eval_file in 'anthropic' 'oasst' 'sinstruct' 'wizardlm' 'vicuna' 'koala'
    do
        k1=sftExp35_llama2-70b_baseline
        k2=$model
        scorer=gpt-4-0613

        echo "==1==pairwise-compare-between-${model}-and-${k1}-on-${eval_file} ...==1=="
        python eval.py -i1 ./results/${k1}/${eval_file}/seed_3517.json -i2 ./results/${k2}/${eval_file}/seed_3517.json -k1 $k1 -k2 $k2 --batch_size 10 --max_tokens 256 --output_dir ./ --eval_scorer $scorer
        echo "==2==pairwise-compare-between-${model}-and-${k1}-on-${eval_file} ...==2=="
        python eval.py -i1 ./results/${k2}/${eval_file}/seed_3517.json -i2 ./results/${k1}/${eval_file}/seed_3517.json -k1 $k2 -k2 $k1 --batch_size 10 --max_tokens 256 --output_dir ./ --eval_scorer $scorer
    done
done

