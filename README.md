# Phased Instruction Fine-Tuning for Large Language Models
This repository provides an overview of all components from the paper Phased Instruction Fine-Tuning for Large Language Models, ACL 2024 Findings.


# Citation
```
@Inproceedings{PhasedSFT,
    author = {Wei Pang and Chuan Zhou  and Xiao-Hua Zhou and Xiaojie Wang},
    title = {Phased Instruction Fine-Tuning for Large Language Models},
    booktitle = {ACL Findings},
    year = {2024},
    pages = {},
}
```


# Code
## bash
run.sh: training bash<br>
stopall.sh: killing<br>


## code dir
generation dir: making inference on 'oasst' 'anthropic' 'koala' 'vicuna' 'sinstruct' 'wizardlm'<br>
evaluation dir: scoring with gpt-4-0613 and then calculating the Win-Rate metric<br>
scripts dir: training scripts<br>
xllm dir: traing codes and dataloader<br>


# Datasets
The instruction difficulty within the Alpaca and Alpaca-cleaned data are quantitatively assessed by GPT-4, assigning scores from 1 to 5, with higher score denoting increased complexity.<br>

## difficulty-stratified instruction dataset
Alpaca-scored: Alpaca 52k dataset scored by the strongest gpt-4-0613, then splited into three stages with difficult increasing.<br>
Alpaca-clean-scored: Alpaca-clean 52k dataset scored by gpt-4-0613 too.<br>


## Alpaca 52K scored by gpt-4-0613
![alpaca_data-gpt-4-0613_v1-score-dist](https://github.com/xubuvd/PhasedSFT/assets/59753505/f93ce7c1-9987-4a54-94d4-ed0455cc1ac2)

## Alpaca-clean 52K scored by gpt-4-0613
![alpaca_data_cleaned-gpt-4-0613_v1-score-dist](https://github.com/xubuvd/PhasedSFT/assets/59753505/bdff903d-0fcd-4ffc-adbf-e9cfebbbc1bf)

