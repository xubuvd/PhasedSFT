#!bin/bash

models=("sftExep8CheckpointStage1")
for model in ${models[*]}
do
    for dataset_name in 'oasst' 'anthropic' 'koala' 'vicuna' 'sinstruct' 'wizardlm'
    do
        echo "infer model ${model} on test dataset of ${dataset_name} ..."
        CUDA_VISIBLE_DEVICES='0,1' python generation_v2.py \
            --model_name_or_path /data/usr/pangwei/${model} \
            --tokenizer_path /data/usr/pangwei/Llama-2-13b-hf \
            --dataset_name ${dataset_name}
    done
done

