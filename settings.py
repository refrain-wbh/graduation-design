
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import random
import numpy as np
def set_seed(seed: int):
    """Sets the relevant random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

from transformers import AdamW


# 将模型的参数分成两组，一组是需要进行权重衰减的参数，另一组是不需要进行权重衰减的参数。
# 这样做的目的是为了避免对一些参数（如偏置和层归一化的权重）进行过度正则化，从而影响模型的性能。
def get_optimizer(model, config_train):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config_train.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=config_train.lr,
                      eps=config_train.adam_epsilon,
                      correct_bias=config_train.bias_correction
    )
    return optimizer

from models.modeliing_bert import BertForSequenceClassification
from models.modeling_roberta import RobertaForSequenceClassification
from transformers import (
    AdamW, AutoConfig, AutoTokenizer,get_linear_schedule_with_warmup 
)
import utils
from torch.utils.data import DataLoader

def get_model_base(model_name):
    if model_name =="bert-base-uncased":
        return BertForSequenceClassification
    elif model_name == "roberta-base":
        return  RobertaForSequenceClassification
    else:
        return BertForSequenceClassification

def get_model_tokenizer(config_model, num_labels):
    model_config = AutoConfig.from_pretrained(config_model.model_name, num_labels=num_labels,mirror='tuna')
    tokenizer = AutoTokenizer.from_pretrained(config_model.model_name)
    model = get_model_base(config_model.model_name).from_pretrained(
                                config_model.model_name, config=model_config)
    model.to(device)
    if config_model.reinit_classifier:
        model.reinit_classifier()
    if config_model.freeze_bert:
        model.freeze_Bert()
    return model, tokenizer

def get_dataloader(tokenizer,config_dataset,split='train',with_idx=False):
    collator = utils.Collator(pad_token_id=tokenizer.pad_token_id,with_idx=with_idx)
    custom_dataset = utils.HuggingfaceDataset(config_dataset= config_dataset, 
                                              tokenizer = tokenizer,
                                              split = split,
                                              with_idx=with_idx)
    
    custom_loader = DataLoader(custom_dataset, 
                               batch_size=config_dataset.batch_size, 
                               shuffle=config_dataset.shuffle, 
                               collate_fn=collator)
    return custom_loader

def get_scheduler(optimizer, config_train, train_loader):
    train_dataset_size = len(train_loader.dataset)
    num_training_steps = train_dataset_size * config_train.epochs // config_train.batch_size
    epoch_steps = train_dataset_size // config_train.batch_size
    warmup_steps = num_training_steps * config_train.warmup_ratio
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_training_steps)
    return scheduler