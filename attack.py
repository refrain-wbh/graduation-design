import os
import csv

import sys
import argparse
from transformers import AutoConfig, AutoTokenizer, BertForSequenceClassification

# from models.modeliing_bert import BertForSequenceClassification

# from modeling_roberta_ER import RobertaForSequenceClassification

from textattack import Attacker
from textattack import AttackArgs

from textattack.models.wrappers import HuggingFaceModelWrapper

# from utils import HuggingfaceDataset
from textattack.datasets import HuggingFaceDataset
from textattack.attack_results import (
    SuccessfulAttackResult,
    MaximizedAttackResult,
    FailedAttackResult,
)
from pathlib import Path
import time
from loguru import logger

cur_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime(time.time() + 8 * 60 * 60))
file_name = Path(__file__).name.split(".")[0]
log_file = f"log/{file_name}/{cur_time}.log"
logger.add(
    log_file,
    rotation="100 MB",
    diagnose=True,
    backtrace=True,
    enqueue=True,
    retention="10 days",
    compression="zip",
)


from build_attacker_utils import build_weak_attacker, build_english_attacker


def do_attack_with_model(
    model, tokenizer, dataset_setting, attack_setting, attack_method
):
    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)
    if attack_method == "bertattack":
        neighbour_vocab_size = 50
        modeify_ratio = 0.9
        sentence_similarity = 0.2
        attack = build_weak_attacker(
            neighbour_vocab_size, sentence_similarity, model_wrapper, attack_method
        )
    elif (
        attack_method == "textfooler" and dataset_setting.name == "SetFit/20_newsgroups"
    ):
        neighbour_vocab_size = 10
        modeify_ratio = 0.9
        sentence_similarity = 0.85
        attack = build_weak_attacker(
            neighbour_vocab_size, sentence_similarity, model_wrapper, attack_method
        )
    else:
        attack = build_english_attacker(model_wrapper, attack_method)
    # attack = build_english_attacker(args, model_wrapper)
    # dataset = utils.Huggingface_dataset(args,tokenizer,dataset_name,
    #                              subset="sst2" if task_name=="sst-2" else task_name
    #                              , split=valid)
    # 注意这里的DataSet是textattack的dataset
    dataset = HuggingFaceDataset(
        name_or_dataset=dataset_setting.name,
        subset=dataset_setting.task_name,
        split="test"
        if dataset_setting.name == "SetFit/20_newsgroups"
        else "validation",
    )
    dataset.shuffle()
    logger.info("shuffled attack set!")
    if dataset_setting.name == "glue" and dataset_setting.task_name == "sst2":
        attack_args = AttackArgs(
            num_examples=attack_setting.num_examples,
            disable_stdout=True,
            random_seed=attack_setting.attack_seed,
            shuffle=False,
        )
    else:
        attack_args = AttackArgs(
            num_examples=attack_setting.num_examples,
            disable_stdout=True,
            random_seed=attack_setting.attack_seed,
            shuffle=True,
        )
    attacker = Attacker(attack, dataset, attack_args)

    num_results = 0
    num_successes = 0
    num_failures = 0

    printed = 0
    for result in attacker.attack_dataset():
        if printed == 0:
            logger.info(result)
            printed += 1
        num_results += 1
        if (
            type(result) == SuccessfulAttackResult
            or type(result) == MaximizedAttackResult
        ):
            num_successes += 1
        if type(result) == FailedAttackResult:
            num_failures += 1

    # compute metric
    original_accuracy = (num_successes + num_failures) * 100.0 / num_results
    accuracy_under_attack = num_failures * 100.0 / num_results
    attack_succ = (
        (original_accuracy - accuracy_under_attack) * 100.0 / original_accuracy
    )

    return {
        "num_successes": num_successes,
        "num_failures": num_failures,
        "num_results": num_results,
        "original_accuracy": original_accuracy,
        "accuracy_under_attack": accuracy_under_attack,
        "attack_succ": attack_succ,
    }
