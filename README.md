# Phased Instruction Fine-Tuning for Large Language Models
该仓库开源了《Phased Instruction Fine-Tuning for Large Language Models》论文中的所有代码和数据，该论文将发表在 ACL 2024 Findings。<br>
This repository provides an overview of all components from the paper Phased Instruction Fine-Tuning for Large Language Models, ACL 2024 Findings.<br>
https://aclanthology.org/2024.findings-acl.341/

# 引用 Citation
```
@Inproceedings{PhasedSFT,
    author = {Wei Pang and Chuan Zhou and Xiao-Hua Zhou and Xiaojie Wang},
    title = {Phased Instruction Fine-Tuning for Large Language Models},
    booktitle = {ACL Findings},
    year = {2024},
    pages = {},
}
```


# 代码 Code
## bash脚本
```
bash run.sh
bash stopall.sh
```

## 代码目录说明 code dir
###1.generation dir: 模型推理，生成测试集的答案 - making inference on 'oasst' 'anthropic' 'koala' 'vicuna' 'sinstruct' 'wizardlm'<br>
```
bash evaluation.sh
```

###2.evaluation dir: 使用gpt-4-0613评测答案的质量，给答案打分 - scoring with gpt-4-0613 and then calculating the Win-Rate metric<br>
```
bash run_gpt4_scoring.sh # gpt-4-0613 打分
bash run_win_rate.sh # 计算胜率
```
<br>

###3.scripts dir: 训练代码的脚本目录 - training scripts<br>

###4.xllm dir: 训练代码和数据加载 - traing codes and dataloader<br>

# 训练数据 - Datasets
Alpaca 和 Alpaca-cleaned 数据中的指令难度通过 GPT-4 进行量化评估，评分范围为 1 到 5 分，分数越高表示复杂性越高。- The instruction difficulty within the Alpaca and Alpaca-cleaned data are quantitatively assessed by GPT-4, assigning scores from 1 to 5, with higher score denoting increased complexity.<br>

## 根据难度分数划分数据集 - difficulty-stratified instruction dataset

Alpaca-scored: Alpaca 52k 数据集由最强版本的 GPT-4-0613 评分后，根据难度递增分为三个阶段。- Alpaca 52k dataset scored by the strongest gpt-4-0613, then splited into three stages with difficult increasing.<br>
Alpaca-clean-scored: Alpaca-clean 52k dataset scored by gpt-4-0613 too.<br>

# 论文总结 - Summary of this paper
![main](https://github.com/xubuvd/PhasedSFT/assets/59753505/4458ed34-241e-43f4-a8b9-161d3c31be03)

上述图表总结了本文的内容。从图中可以看出，随着逐步训练的推进，五个大型语言模型（LLMs）在难度递增的多阶段子数据集上的胜率呈增长趋势（实线）。这一趋势与同样五个大型语言模型在难度随机分布的多阶段子数据集上的胜率趋势（虚线）形成了鲜明对比。 - The above Figure is a summary of this paper. As can be seen from the figure, with the progression of uptraining, it demonstrates the winning rate growth trend （the solid line） of five LLMs on multi-stage sub-datasets with increasing difficulty. This forms a stark contrast to the winning rate trend of the same five LLMs on multi-stage sub-datasets with randomly distributed difficulty levels （the dotted-line）.<br>

## Alpaca 52K scored by gpt-4-0613
![alpaca_data-gpt-4-0613_v1-score-dist](https://github.com/xubuvd/PhasedSFT/assets/59753505/f93ce7c1-9987-4a54-94d4-ed0455cc1ac2)

## Alpaca-clean 52K scored by gpt-4-0613
![alpaca_data_cleaned-gpt-4-0613_v1-score-dist](https://github.com/xubuvd/PhasedSFT/assets/59753505/bdff903d-0fcd-4ffc-adbf-e9cfebbbc1bf)

