import os
import json
from typing import *
import torch
from torch.utils.data import IterableDataset, Dataset
from tqdm import tqdm
from transformers.tokenization_utils import PreTrainedTokenizer
import copy
import random

IGNORE_INDEX=-100

def load_single_file(data_file):
    with open(data_file)as f:
        lines = f.readlines()
    return [json.loads(l) for l in lines]

def load_raw_data(data_dir, max_sample=None, random_state=0):
    
    raw_dataset = []
    
    for f_ in os.listdir(data_dir):
        if f_.endswith("json") and "part1" in f_:
            print("load part1 data only")
            f_ = os.path.join(data_dir, f_)
            raw_dataset += load_single_file(f_)
    
    if max_sample is not None:
        random.seed(random_state)
        raw_dataset = list(random.sample(raw_dataset, max_sample))
    
    return raw_dataset

def check_alternate_human_gpt(conv):
    
    length = len(conv)
    
    if len(conv) % 2 != 0:
        print(conv)
        return False
    tags = [i for _ in range(len(conv)//2) for i in ["human", "gpt"]]
    
    for i in range(len(conv)):
        if tags[i] != conv[i]["from"]:
            print(conv)
            return False
    return True

def load_sharegpt_data(data_file):
    new_data = []
    data = json.load(open(data_file, "r"))
    for item in data:
        conv = item["conversations"]
        if conv[0]["from"] != "human":
            conv = conv[1:]
        if conv[-1]["from"] != "gpt":
            conv = conv[:-1]
        if check_alternate_human_gpt(conv):
            data = {"id": item["id"], "data": [c["value"] for c in conv]}
            new_data.append(data)
    return new_data

def load_reasoning_data(data_dir, guid=0):
    new_data = []
    guid = guid
    for f_ in os.listdir(data_dir):
        if f_.endswith("json") or f_.endswith("jsonl"):
            if f_.endswith("json"):
                data = json.load(open(os.path.join(data_dir, f_), "r", encoding="utf-8"))
            else:
                data = [json.loads(l) for l in open(os.path.join(data_dir, f_), "r", encoding="utf-8").readlines()]
            for item in data:
                data = {"id": f"reasoning-{guid}", "data": [item["question"].replace(u'\xa0', u' '), item["answer"].replace(u'\xa0', u' ')]}
                new_data.append(data)
                guid += 1
    return new_data

def load_zh_data(datafile):
    new_data = []
    guid = 0
    with open(datafile)as f:
        lines = f.readlines()
        for l in lines:
            d = json.loads(l)
            conv = []
            i = 1
            while f"question_{i}" in d:
                conv.append(d[f"question_{i}"].replace(u'\xa0', u' ').strip())
                conv.append(d[f"answer_{i}"].replace(u'\xa0', u' ').strip())
                i += 1

            data = {"id": f"zh-{guid}", "data": conv}
            new_data.append(data)
            guid += 1
    return new_data
    

def collator(tokenizer, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
    input_ids, labels, attention_mask = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "attention_mask"))
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels)
    attention_mask = torch.stack(attention_mask)
    # input_ids = torch.nn.utils.rnn.pad_sequence(
    #     input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    # )
    # labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
    return dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
    )


class PromptIterableDataset(IterableDataset):
    def __init__(self,
                 raw_dataset: Union[Dataset, List],
                 sep: List = ["EOS", "\n"],
                 tokenizer: PreTrainedTokenizer = None,
                 max_seq_length: Optional[int] = 512,
                 teacher_forcing: Optional[bool] = True,
                 truncate_method: Optional[str] = "tail",
                ):
        assert hasattr(raw_dataset, "__iter__"), f"The dataset must have __iter__ method. dataset is {raw_dataset}"
        assert hasattr(raw_dataset, "__len__"), f"The dataset must have __len__ method. dataset is {raw_dataset}"
        self.raw_dataset = raw_dataset
        self.sep = sep
        self._end_token = None
        self.start_token = self.sep[-1]
        self.teacher_forcing = teacher_forcing
        assert self.teacher_forcing, print("must use teacher forcing")

        self.tokenizer = tokenizer
        self.truncate_method = truncate_method
        self.max_seq_length = max_seq_length
        assert self.truncate_method == "tail", print("only tail truncate support")
    
    @property
    def end_token(self):
        if self._end_token is not None:
            return self._end_token
        end_token = self.sep[0]
        if end_token == "EOS":
            self._end_token = self.tokenizer.eos_token
        else:
            self._end_token = end_token
        return self._end_token

    def tokenize_example(self, example):
        end_token = self.end_token
        tags = [i for _ in range(len(example["data"])//2) for i in ["User", "Assistant"]]
        if str(example["id"]).startswith("reasoning-"):
            assert len(example["data"]) == 2, print(example)
            tags = ["Question", "Answer"]
        labels = []
        tokenized_ids = []
        for i, c in enumerate(example["data"]):
            c_new = tags[i] + ": " + c + end_token
            if i % 2 == 1:
                # model
                c_input = self.start_token + tags[i] + ": "
                tokenized = self.tokenizer(c_input, add_special_tokens=False)
                tokenized_ids += tokenized["input_ids"]
                labels += [IGNORE_INDEX] * len(tokenized["input_ids"])

                c_generate = c + end_token
                tokenized = self.tokenizer(c_generate, add_special_tokens=False)
                tokenized_ids += tokenized["input_ids"]
                labels += tokenized["input_ids"]
            else:
                # user
                if i == 0:
                    # no start token
                    c_new = self.tokenizer.bos_token + tags[i] + ": " + c + end_token
                else:
                    c_new = self.start_token + tags[i] + ": " + c + end_token
                tokenized = self.tokenizer(c_new, add_special_tokens=False)
                tokenized_ids += tokenized["input_ids"]
                labels += [IGNORE_INDEX] * len(tokenized["input_ids"])

        assert len(tokenized_ids) == len(labels)

        return {"input_ids": torch.LongTensor(tokenized_ids), "labels": torch.LongTensor(labels)}

    def pad_truncate(self, tokenized_example):
        old_len = len(tokenized_example["input_ids"])
        tokenized_example["attention_mask"] = torch.LongTensor([1]*len(tokenized_example["input_ids"]))
        if old_len > self.max_seq_length:
            for k in tokenized_example:
                tokenized_example[k] = tokenized_example[k][:-(old_len - self.max_seq_length)]
        elif old_len < self.max_seq_length:
            tokenized_example["input_ids"] = torch.cat([torch.LongTensor([self.tokenizer.pad_token_id]*(self.max_seq_length - old_len)), tokenized_example["input_ids"]])
            tokenized_example["labels"] = torch.cat([torch.LongTensor([IGNORE_INDEX]*(self.max_seq_length - old_len)), tokenized_example["labels"]])
            tokenized_example["attention_mask"] = torch.LongTensor([0]*(self.max_seq_length - old_len) + [1]*old_len)
        assert len(tokenized_example["input_ids"]) == len(tokenized_example["labels"]) == len(tokenized_example["attention_mask"]) == self.max_seq_length
        return tokenized_example

    def __iter__(self):
        for example in self.raw_dataset:
            tokenized_example = self.tokenize_example(example)
            tokenized_example = self.pad_truncate(tokenized_example)
            yield tokenized_example

    def __len__(self):
        return len(self.raw_dataset)


if __name__ == "__main__":
    from transformers import AutoTokenizer, LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained("")
    print("loading...")
    raw_datasets = load_single_file("")
    print("done")
    dataset = PromptIterableDataset(raw_datasets, tokenizer=tokenizer, max_seq_length=2048, teacher_forcing=True)

    from IPython import embed; embed()

    for data in dataset:
        print(data)
        print(tokenizer.decode(data["input_ids"][:1000]))
        
        model_output = data["input_ids"][:1000][data["labels"][:1000]!=-100]
        print("##### model output")
        print(tokenizer.decode(model_output))
        break