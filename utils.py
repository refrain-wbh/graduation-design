import ruamel.yaml

yaml = ruamel.yaml.YAML()
from easydict import EasyDict as edict
import copy


# 定义一个函数，递归更新字典,target更新到base中去
def update_dict(base, target):
    for key, value in target.items():
        if isinstance(value, dict):
            base[key] = update_dict(base.get(key, {}), value)
        else:
            base[key] = value
    return base


# 将target中有或者修改过的key-value找出来
def sub_dict(target, base):
    result = {}
    for key, value in target.items():
        if isinstance(value, dict):
            _t = sub_dict(value, base.get(key, {}))
            if _t != {}:
                result[key] = _t
        elif key not in base.keys() or base[key] != value:
            result[key] = value
    return result


def read_config(config_path="config/default.yaml"):
    default_config = edict(
        yaml.load(open("config/default.yaml", "r", encoding="utf-8"))
    )
    custom_config = edict(yaml.load(open(config_path, "r", encoding="utf-8")))
    return update_dict(default_config, custom_config)


def read_configs(config_path):
    default_config = edict(
        yaml.load(open("config/default.yaml", "r", encoding="utf-8"))
    )
    config_loader = list(yaml.load_all(open(config_path, "r", encoding="utf-8")))
    config_loader = [
        edict(update_dict(copy.deepcopy(default_config), edict(config)))
        for config in config_loader
    ]
    return config_loader


def write_config(custom_config, config_path):
    default_config = edict(
        yaml.load(open("config/default.yaml", "r", encoding="utf-8"))
    )
    sub_config = sub_dict(custom_config, default_config)
    if os.path.exists(config_path):
        old_config = list(yaml.load_all(open(config_path, "r", encoding="utf-8")))
        old_config.append(sub_config)
        yaml.dump_all(old_config, open(config_path, "w", encoding="utf-8"))
    else:
        yaml.dump(sub_config, open(config_path, "w", encoding="utf-8"))


import random
import numpy as np
import torch

import settings

from torch.optim.lr_scheduler import LambdaLR


def get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, last_epoch=-1
):
    """Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    From:
        https://github.com/uds-lsv/bert-stable-fine-tuning/blob/master/src/transformers/optimization.py
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


from torch.nn.utils.rnn import pad_sequence


def pad_squeeze_sequence(sequence, *args, **kwargs):
    """Squeezes fake batch dimension added by tokenizer before padding sequence."""
    return pad_sequence([x.squeeze(0) for x in sequence], *args, **kwargs)


class Collator:
    def __init__(self, pad_token_id=0, with_idx=False):
        self._pad_token_id = pad_token_id
        self.with_idx = with_idx

    def __call__(self, features):
        # 输入和标签分离
        if self.with_idx:
            model_inputs, labels, idx = list(zip(*features))
        else:
            model_inputs, labels = list(zip(*features))
        # 假定所有的model_inputs都是一样的格式
        proto_input = model_inputs[0]
        keys = list(proto_input.keys())
        padded_inputs = {}
        # 对于每个key，将其对应的value进行pad
        for key in keys:
            # 需要对input_ids进行pad，其他的填充0
            padding_value = self._pad_token_id if key == "input_ids" else 0
            # 去掉batch维度
            sequence = [x[key] for x in model_inputs]
            padded = pad_squeeze_sequence(
                sequence, batch_first=True, padding_value=padding_value
            )
            padded_inputs[key] = padded
        labels = torch.tensor(labels)
        # padded_inputs = {k: v.to(settings.device) for k, v in padded_inputs.items()}
        # labels = labels.to(settings.device)
        if self.with_idx:
            return padded_inputs, labels, idx
        else:
            return padded_inputs, labels


task_to_keys = {
    "ag_news": ("text", None),  # 新闻分类任务，用来根据新闻标题和描述将新闻分为四个类别
    "imdb": ("text", None),  # 电影评论情感分析任务，用来根据评论内容判断评论的情感是正面还是负面。
    "cola": ("sentence", None),  # 句子语法正确性判断任务，用来判断句子是否符合语法规则。
    "mnli": ("premise", "hypothesis"),  # 自然语言推理任务，给出一个前提句和一个假设句，判断假设句是否可以由前提句推导出来。
    "mrpc": ("sentence1", "sentence2"),  # 语义相似度判断任务，给出两个句子，判断它们之间的语义是否相似。
    "qnli": ("question", "sentence"),  # 自然语言推理任务，给出一个问题和一个句子，判断问题是否可以由句子回答。
    "qqp": ("question1", "question2"),  # 语义相似度判断任务，给出两个问题，判断它们之间的语义是否相似。
    "rte": ("sentence1", "sentence2"),  # 自然语言推理任务，给出一个前提句和一个假设句，判断假设句是否可以由前提句推导出来。
    "sst2": ("sentence", None),  # 句子情感分析任务，用来判断句子的情感是正面还是负面。
    "stsb": ("sentence1", "sentence2"),  # 语义相似度判断任务，给出两个句子，判断它们之间的语义是否相似。
    "wnli": ("sentence1", "sentence2"),  # 自然语言推理任务，给出一个前提句和一个假设句，判断假设句是否可以由前提句推导出来。
    "SetFit/20_newsgroups": ("text", None),  # 新闻分类任务，用来根据新闻标题和描述将新闻分为四个类别
}
from torch.utils.data import Dataset
import datasets
import os


class HuggingfaceDataset(Dataset):
    def __init__(
        self,
        config_dataset,
        tokenizer,
        split,
        with_idx=False,
    ):
        self.config = config_dataset
        self.tokenizer = tokenizer
        self.with_idx = with_idx
        self.name = self.config.name
        self.task_name = (
            self.config.task_name if hasattr(self.config, "task_name") else None
        )
        self.dataset = datasets.load_dataset(self.name, self.task_name, split=split)

        if self.with_idx and "idx" not in self.dataset.data.schema.names:
            self.dataset.data.add_column("idx", [i for i in range(len(self.dataset))])
        # 不应该在这里shuffle，因为在dataloader中会shuffle

    def get_column(self, column_name):
        return self.dataset.data[column_name]

    def random_select(self):
        example_num = (
            self.config.num_train_examples_ratio
            * len(self.dataset)
            // self.config.num_labels
        )
        unselect_label_nums = {
            label: example_num for label in range(self.args.num_labels)
        }
        selected_indexs = []
        # 随机抽取数据集的下标，注意与数据集本身的idx字段不同
        for one_index in np.random.permutation(range(len(self.dataset))):
            label = self.dataset[one_index]["label"]
            if unselect_label_nums[label] > 0:
                selected_indexs.append(one_index)
                unselect_label_nums[label] -= 1
            if sum(unselect_label_nums.values()) == 0:
                break
        return self.dataset.select(selected_indexs)

    def _format_examples(self, examples):
        key1, key2 = task_to_keys[
            self.name if self.task_name is None else self.task_name
        ]
        texts = (examples[key1],) if key2 is None else (examples[key1], examples[key2])
        inputs = self.tokenizer(
            *texts,
            truncation=True,
            max_length=self.config.max_seq_length,
            return_tensors="pt"
        )

        if self.with_idx:
            return (inputs, int(examples["label"]), examples["idx"])
        else:
            return (inputs, int(examples["label"]))

    def shuffle(self):
        self.dataset.shuffle()

    def __len__(self):
        return len(self.dataset)

    # 一个三元组，分别是input，label，idx
    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self._format_examples(self.dataset[idx])
        else:
            return [
                self._format_examples(self.dataset[j])
                for j in range(idx.start, idx.stop)
            ]


class ExponentialMovingAverage:
    def __init__(self, weight=0.3):
        self._weight = weight
        self.reset()

    def update(self, x, i=1):
        self._x += x
        self._i += i

    def reset(self):
        self._x = 0
        self._i = 0

    def get_metric(self):
        return self._x / (self._i + 1e-13)
