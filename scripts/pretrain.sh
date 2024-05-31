#!/bin/bash

set -e

log_out=0
only_print=0
dist_only_print=0
debug_nccl=0
debug_nsys=0
debug_dataloader="False"
enable_flash_attn="True"
train_embed="False"
tie_embed="False"

nodes=
#master_addr=
gpu_per_node=8

datestr=`date +"%Y-%m-%d"`

wandb_run_name="sftest-$datestr"
echo "wandb_run_name=$wandb_run_name"

data_path=/data/usr/pangwei/frontllm/sft/paper/sft_train5_random_ablation

data_suffix=".jsonl"
output_path=/data/usr/pangwei/XXXXX
echo "data_suffix=$data_suffix"
echo "output_path=$output_path"

ckpt_path=/data/usr/pangwei/Llama-2-13b-hf
echo "ckpt_path=$ckpt_path"

tokenizer_path=/data/usr/pangwei/Llama-2-13b-hf
#/data/models/llama-2/llama-2-70b-hf
#/data/usr/pangwei/Llama-2-13b-hf

strategy=zero3
bs_per_dev=16
#16 for llama-13b
#4 for llama-70b

# save model per 500 global_step
ckpt_steps=100000
train_epoch=20

lr="5e-6"
warmup_ratio=0.1
#warmup_steps=162

# while [[ "$1" =~ ^- && ! "$1" == "--" ]]; do
while [[ $# -gt 0 ]]; do
	case $1 in
	-h | -H | --help)
		cat <<HelpMessage
  Pretrain models with FrontLLM.
    -h/-H/--help: print help messages
HelpMessage
		exit
		;;
	-n | --nodes)
		shift
		nodes=$1
		;;
	-g | --gpu-num)
		shift
		gpu_per_node=$1
		;;
	-s | --strategy)
		shift
		strategy=$1
		;;
	--data)
		shift
		data_path=$1
		;;
	--ckpt)
		shift
		ckpt_path=$1
		;;
	--output)
		shift
		output_path=$1
		;;
	-b | --batch-size-per-device)
		shift
		bs_per_dev=$1
		;;
	--train-steps)
		shift
		train_steps=$1
		;;
	--ckpt-steps)
		shift
		ckpt_steps=$1
		;;
	--lr)
		shift
		lr=$1
		;;
	--warmup-ratio)
		shift
		warmup_ratio=$1
		;;
	--warmup-steps)
		shift
		warmup_steps=$1
		;;
	--wandb-run-name)
		shift
		wandb_run_name="$1"
		;;
	-p | --only-print)
		only_print=1
		;;
	-P | --dist-only-print)
		dist_only_print=1
		;;
	-l | -L | --log-out)
		log_out=1
		;;
	--disable-flash-attn)
		enable_flash_attn="False"
		;;
	--train-embed)
		train_embed="True"
		;;
	--tie-embed)
		tie_embed="True"
		;;
	--debug-nccl)
		debug_nccl=1
		;;
	--debug-nsys)
		debug_nsys=1
		;;
	--debug-data)
		debug_dataloader="True"
		;;
	*)
		echo "ERORR: Unsupported flag [$1]"
		exit
		;;
	esac
	shift
done
if [[ "$1" == '--' ]]; then shift; fi

# Parse Args

framework=""
if [ $strategy == "full_shard" ] || [ $strategy == "shard_grad_op" ]; then
	framework="fsdp"
	echo "===== Use Pytorch FSDP strategy: [$strategy] ====="
elif [ $strategy == "zero2" ] || [ $strategy == "zero3" ]; then
	framework="deepspeed"
	echo "===== Use DeepSpeed strategy: [$strategy] ====="
else
	echo "ERROR: --strategy should be one of 'full_shard', 'shard_grad_op'(for Pytorch FSDP), 'zero2', 'zero3'(for DeepSpeed)"
	exit
fi

REPO=$(pwd)
config=$REPO/scripts/${framework}.${strategy}.json

# Run Command

CMD=""

CMD="$CMD PYTHONPATH=$REPO"

if [[ $debug_nccl == "1" ]]; then
	set +e
	rm nccl.*.log
	set -e
	CMD="$CMD NCCL_DEBUG=info NCCL_DEBUG_FILE='/localdisk/logs/nccl.%h.%p.log'"
fi

if [[ -z $nodes ]]; then
	if [[ $debug_nsys == "1" ]]; then
		CMD="$CMD nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu --osrt-threshold=10000"
		CMD="$CMD -o nsight_report -f true"
		CMD="$CMD --capture-range=cudaProfilerApi --capture-range-end=stop"
	fi
	CMD="$CMD /root/miniconda3/envs/PiT/bin/torchrun"
else
	CMD="$CMD /data/usr/pangwei/frontllm/sft/paper/frontllm/launch/dist_torchrun --nodes $nodes"
	if [[ $debug_nsys == "1" ]]; then
		CMD="$CMD --debug-nsys"
	fi
	if [[ $dist_only_print == "1" ]]; then
		CMD="$CMD --only-print"
	fi
fi

#CMD="$CMD --master_addr $master_addr"

# to keep total_batch_size equals to 4m (which is what llama got),
# we must keep grad_acc_step * bs_per_dev  = 4m / (2048 * 16 * 8)
grad_acc_step=$((16 / $bs_per_dev))
echo "grad_acc_step=" $grad_acc_step
CMD="$CMD --nproc_per_node $gpu_per_node"
CMD="$CMD $REPO/frontllm/pretrain.py"
CMD="$CMD --model_name_or_path $ckpt_path --data_path $data_path --data_suffix $data_suffix --output_dir $output_path --tokenizer_path $tokenizer_path"
CMD="$CMD --per_device_train_batch_size $bs_per_dev --gradient_accumulation_steps $grad_acc_step"

echo $CMD

# hyper paramters from https://xianyuan.feishu.cn/docx/IPNDdCA5XoHcZAxEjuOcg55pn9b
CMD="$CMD --learning_rate $lr --weight_decay 0.1"
if [[ -n $warmup_ratio ]]; then
	CMD="$CMD --warmup_ratio $warmup_ratio"
else
	CMD="$CMD --warmup_steps $warmup_steps"
fi
CMD="$CMD --adam_beta1 0.9 --adam_beta2 0.95 --max_grad_norm 1.0 --lr_scheduler_type 'cosine'"

CMD="$CMD --bf16 True --tf32 True"
CMD="$CMD --evaluation_strategy 'no'"

CMD="$CMD --save_strategy 'steps' --save_steps $ckpt_steps --save_total_limit 20"
#CMD="$CMD --max_steps $train_steps"
CMD="$CMD --num_train_epochs $train_epoch"

CMD="$CMD --logging_steps 1"
CMD="$CMD --gradient_checkpointing True"

if [[ -n $wandb_run_name ]]; then
	export WANDB_PROJECT=FrontLLM
	CMD="$CMD --report_to wandb --run_name $wandb_run_name"
else
	CMD="$CMD --report_to none"
fi

if [[ $framework == "fsdp" ]]; then
	CMD="$CMD --fsdp '${strategy} auto_wrap' --fsdp_config ${config} --gradient_checkpointing True"
else
	CMD="$CMD --deepspeed ${config}"
fi

if [[ $log_out == "1" ]]; then
	log_file="/localdisk/logs/script.pretrain.log"
	CMD="$CMD 1>${log_file} 2>&1"
fi

# FrontLLM args
CMD="$CMD --enable_flash_attn $enable_flash_attn"
CMD="$CMD --only_train_embedding $train_embed --tie_word_embeddings $tie_embed"
CMD="$CMD --only_debug_dataload $debug_dataloader"

printf "===== Running Command =====\n"
printf "\t%s\n\n" "$CMD"

if [[ $only_print == "0" ]]; then
	printf "===== Command Logs =====\n"
	if [[ $log_out == "1" ]]; then
		echo "Command is running...."
		echo "Please run [tail -f ${log_file}] in another shell to monitoring the running process."
	fi
	if [[ -d /localdisk/logs ]]; then
		timestamp=$(date +"%Y%m%d.%H.%M.%S")
		mv /localdisk/logs /localdisk/logs.$timestamp
	fi
	mkdir /localdisk/logs
	eval "$CMD"
fi

