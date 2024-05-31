#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import pathlib
import socket
import sys
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import torch
import transformers
from loguru import logger
from torch.utils.data import DataLoader, IterableDataset
from transformers import LlamaForCausalLM, Trainer
from transformers import set_seed

from xllm.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
from xllm.packed_dataset import CombinedDataset, PackedDataset

from xllm_dataset import PromptIterableDataset
import json
import random
from transformers import LlamaTokenizer

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    tokenizer_path: Optional[str] = field(default="facebook/opt-125m")

@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    data_suffix : str = field(
        default=None, metadata={"help": "Path to the trained file to be loaded"}
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. "
            + "Sequences will be right padded (and possibly truncated)."
        },
    )


@dataclass
class xLLMArguments:
    enable_flash_attn: bool = field(default=True)
    tie_word_embeddings: bool = field(default=False)

    only_train_embedding: Optional[bool] = field(default=False)
    only_debug_dataload: Optional[bool] = field(default=False)


class xLLMTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self._train_batch_size,
            shuffle=False,
            pin_memory=True,
        )

# Random seeds for reproducability
def set_random_seed(seed=None):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_dataset(
        data_path, 
        tokenizer, 
        max_seq_length=2048,
        data_suffix=None
        ) -> IterableDataset:

    def load_single_file(data_file,local_rank):
        #print(f"local_rank:{local_rank}, data_file:{data_file}")
        with open(data_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        return [json.loads(l) for l in lines]

    def load_raw_data(data_dir, max_sample=None, random_state=0,local_rank=-1,data_suffix=None):
        raw_dataset = []
        for f_ in os.listdir(data_dir):
            if f_.endswith(data_suffix):#("jsonl"):
                f_ = os.path.join(data_dir, f_)
                raw_dataset += load_single_file(f_,local_rank)
        if max_sample is None:
            max_sample = len(raw_dataset)
        random.seed(random_state)
        raw_dataset = list(random.sample(raw_dataset, max_sample))
        return raw_dataset
    
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["RANK"])
    print(f"world_size:{world_size}, local_rank:{local_rank}")

    raw_datasets = load_raw_data(data_dir=data_path,local_rank=local_rank,data_suffix=data_suffix)
    print(f"data_suffix:{data_suffix}, local_rank:{local_rank}, raw_datasets:{len(raw_datasets)}")

    datasize_per_gpu = len(raw_datasets) // world_size
    raw_datasets = raw_datasets[local_rank * datasize_per_gpu : (local_rank + 1) * datasize_per_gpu]

    dataset = PromptIterableDataset(raw_datasets, tokenizer = tokenizer, max_seq_length = max_seq_length, teacher_forcing=True, truncate_method="tail")
    return dataset

def test_dataset(dataset: IterableDataset):
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=8,
        shuffle=False,
        pin_memory=True,
    )
    for i, item in enumerate(dataloader):
        if i == 20:
            break
        logger.debug(f"{i=}, {item=}")


def train():
    '''
    PYTHONPATH=/localdisk/frontllm frontllm/launch/dist_torchrun --nodes 1 --nproc_per_node 7 /localdisk/frontllm/frontllm/pretrain.py --model_name_or_path /localdisk/pretrain_ckpt/checkpoint-76500 --data_path /localdisk/sft_train --output_dir /localdisk/sft_ckpt --per_device_train_batch_size 16 --gradient_accumulation_steps 1 --learning_rate 2e-4 --weight_decay 0.1 --warmup_steps 2000 --adam_beta1 0.9 --adam_beta2 0.95 --max_grad_norm 1.0 --lr_scheduler_type 'cosine' --bf16 True --tf32 True --evaluation_strategy 'no' --save_strategy 'steps' --save_steps 500 --save_total_limit 20 --max_steps 76800 --logging_steps 1 --gradient_checkpointing True --report_to wandb --run_name sft --deepspeed /localdisk/frontllm/scripts/deepspeed.zero2.json --enable_flash_attn True --only_train_embedding False --tie_word_embeddings False --only_debug_dataload False
    '''
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, xLLMArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        xllm_args,
    ) = parser.parse_args_into_dataclasses()
    logger.info(xllm_args)

    if xllm_args.enable_flash_attn:
        replace_llama_attn_with_flash_attn()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.tokenizer_path)
    #trust_remote_code=True)
    #LlamaTokenizer.from_pretrained(model_args.tokenizer_path)
    tokenizer.pad_token_id = 0

    # train_dataset = get_dataset(data_path=data_args.data_path)
    train_dataset = get_dataset(data_path=data_args.data_path, 
                                    tokenizer=tokenizer,
                                    max_seq_length=2048,
                                    data_suffix=data_args.data_suffix)

    if xllm_args.only_debug_dataload:
        test_dataset(train_dataset)
        return
    print(f"data_path:{data_args.data_path}/{data_args.data_suffix}")
    print(f"model_name_or_path={model_args.model_name_or_path}")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        #trust_remote_code=True
    )
    if not isinstance(model, LlamaForCausalLM):
        raise ValueError("We only support LlamaForCausalLM now")

    model.config.pad_token_id = 0
    model.config.use_cache = False
    model.config.tie_word_embeddings = xllm_args.tie_word_embeddings
    logger.warning(f"{model.config.tie_word_embeddings=}")

    # NEW_VOCAB_SIZE = 100261  # gpt-4 tokenizer
    NEW_VOCAB_SIZE = tokenizer.vocab_size #49953  # cn-llama tokenizer
    print(f"tokenizer.vocab_size:{NEW_VOCAB_SIZE}")
    model.resize_token_embeddings(NEW_VOCAB_SIZE)

    if xllm_args.only_train_embedding:
        logger.warning("Only train input/output_embedding with other params freezed")
        for name, param in model.named_parameters():
            param.requires_grad = False
        for name, param in model.get_input_embeddings().named_parameters():
            param.requires_grad = True
        for name, param in model.get_output_embeddings().named_parameters():
            param.requires_grad = True

        trainable_param_cnt = sum(
            1 for para in model.parameters() if para.requires_grad is True
        )
        assert trainable_param_cnt == 2, f"Wrong number of {trainable_param_cnt=}"

    trainer = xLLMTrainer(
        model=model,
        # tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
    )
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

if __name__ == "__main__":
    global_rank = os.environ["RANK"]
    ip = socket.gethostbyname(socket.gethostname())
    logger.remove()
    logger.add(sys.stdout, level="INFO", colorize=True)
    logger.add(
        f"./logs/pretrain.{ip}.rank_{global_rank}.log", level="DEBUG", colorize=True
    )
    set_random_seed(seed=3517)

    logger.info("Start pretrain")
    train()

