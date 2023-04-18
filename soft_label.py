import argparse
import logging
import os
from pathlib import Path
import random
import numpy as np
from tqdm import tqdm
import sys

os.environ["HF_DATASETS_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from transformers import AdamW, AutoConfig, AutoTokenizer, BertForSequenceClassification

# from models.modeliing_bert import BertForSequenceClassification
# from models.modeling_roberta import RobertaForSequenceClassification
from torch.utils.tensorboard import SummaryWriter

import utils, settings, select_data

# from transformers.models.bert.modeling_bert import BertSelfAttention, BertLayer # modified
from transformers.models.bert.modeling_bert import (
    BertSelfAttention,
    BertLayer,
)  # modified
import time
from loguru import logger
from pathlib import Path

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


def SoftCrossEntropy(inputs, target, reduction="none", soft_label=1, device="cuda"):
    """
    soft label的loss实现
    :param inputs:
    :param target:
    :param reduction:
    :param soft_label: golden label的值，剩下的值被其他标签平分
    :param device:
    :return:
    """
    log_likelihood = -F.log_softmax(inputs, dim=1)
    num_labels = inputs.shape[1]
    batch = inputs.shape[0]

    new_target = F.one_hot(target, num_labels).to(device)
    # 非golden位置为0
    inverse_target = (torch.ones(inputs.shape).to(device) - new_target).to(device)

    new_target = new_target * soft_label + inverse_target * (
        (1 - soft_label) / (num_labels - 1)
    )
    losses = torch.sum(torch.mul(log_likelihood, new_target), dim=1)
    if reduction == "average":
        losses = torch.sum(losses) / batch
    elif reduction == "none":
        return losses
    elif reduction == "sum":
        losses = torch.sum(losses)

    return losses


def evaluate(model, dev_loader):
    logger.info("Evaluating...")
    model.eval()
    correct = 0
    total = 0
    avg_loss = utils.ExponentialMovingAverage()
    with torch.no_grad():
        for model_inputs, labels in dev_loader:
            logits = model(**model_inputs).logits
            losses = F.cross_entropy(logits, labels)
            loss = torch.mean(losses)
            avg_loss.update(loss.item())

            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        accuracy = correct / (total + 1e-13)
    logger.info(f"Clean Aua: {accuracy}")
    logger.info(f"Clean Loss: {avg_loss.get_metric()}")
    return accuracy, avg_loss
    # logger.info(f'Best dev metric: {best_accuracy} in Epoch: {best_dev_epoch}')


@logger.catch
def soft_label_train(config):
    # tensorboard_path = "/root/workspace/Robust-Data/runs/new_finetune_soft_label_from_init_not_symmetry"+args.output_dir[args.output_dir.rfind("/"):]
    # writer = SummaryWriter(tensorboard_path)
    settings.set_seed(config.seed)

    dataset_setting = config.dataset_setting
    train_setting = config.train_setting
    model_setting = config.model_setting
    statistics_setting = config.statistics_setting
    attack_setting = config.attack_setting
    select_setting = config.select_data
    adverisal_setting = config.adverisal_setting

    model, tokenizer = settings.get_model_tokenizer(
        model_setting, dataset_setting.num_labels
    )
    optimizer = settings.get_optimizer(model, train_setting)
    train_loader = settings.get_dataloader(
        tokenizer, dataset_setting, "train", with_idx=True
    )

    scheduler = settings.get_scheduler(optimizer, train_setting, train_loader)

    valid_loader = settings.get_dataloader(tokenizer, dataset_setting, "validation")
    if train_setting.do_test:
        test_loader = settings.get_dataloader(tokenizer, dataset_setting, "test")

    df = select_data.get_df_from_statistics_file(select_setting.statistics_source)

    result_data_indices, selected_label_nums = select_data.generate_data_indices(
        select_setting, df, dataset_setting.num_labels
    )
    selected_set = set(result_data_indices)
    logger.debug(f"selected_label_nums:{selected_label_nums}")

    one_epoch_steps = int(len(train_loader.dataset) // train_setting.batch_size)
    # if args.save_steps < 1 and args.save_steps > 0:
    #    args.save_steps = int(one_epoch_steps * args.save_steps)
    # save_steps = args.save_steps

    best_accuracy = 0
    global_step = 0

    for epoch in range(train_setting.epochs):
        avg_loss = utils.ExponentialMovingAverage()
        model.train()
        pbar = tqdm(train_loader, desc="Epoch {}".format(epoch))
        """
        for model_inputs, labels, indices in pbar:
            data_selected = torch.tensor(
                [1 if idx in selected_set else 0 for idx in indices]
            )
            data_not_selected = torch.tensor(
                [0 if idx in selected_set else 1 for idx in indices]
            )
            batch_loss = 0
            logits = model(**model_inputs).logits
            preds = logits.argmax(dim=-1)

            losses_hard = F.cross_entropy(logits, labels.squeeze(-1), reduction="none")
            losses_soft = SoftCrossEntropy(
                logits,
                labels.squeeze(-1),
                reduction="none",
                soft_label=train_setting.soft_label,
                device=settings.device,
            )

            soft_coef = train_setting.beta * (torch.ones_like(losses_soft))
            hard_loss = data_not_selected.to(settings.device).mul(losses_hard)
            soft_loss = (
                data_selected.to(settings.device)
                .mul(soft_coef.to(settings.device))
                .mul(losses_soft)
            )
            losses = soft_loss + hard_loss
            loss = torch.mean(losses)

            model.zero_grad()
            loss.backward()

            # 梯度裁剪，避免出现梯度爆炸情况
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            avg_loss.update(batch_loss)

            # writer.add_scalars(
            #    "train_loss", {"whole_loss": avg_loss.get_metric()}, global_step
            # )

            # args.writer.add_scalars("train_loss/whole_loss", avg_loss.get_metric(), global_step=global_step, walltime=None)

            pbar.set_description(
                f"epoch: {epoch: d}, "
                f"loss: {avg_loss.get_metric(): 0.4f}, "
                f'lr: {optimizer.param_groups[0]["lr"]: .3e},'
                # f'non_robust_soft:{non_robust_soft_loss_avg_all.get_metric():0.4f},'
                # f'non_robust_hard:{non_robust_hard_loss_avg_all.get_metric():0.4f},'
                # f'robust_soft:{robust_soft_loss_avg_all.get_metric():0.4f},'
                # f'robust_hard:{robust_hard_loss_avg_all.get_metric():0.4f},'
            )
            global_step += 1
        """
        # valid数据集上验证
        if train_setting.do_eval:
            accuracy, clean_loss = evaluate(model, valid_loader)
            logger.info(
                f"Epoch: {epoch}, "
                f"Loss: {avg_loss.get_metric(): 0.4f}, "
                f'Lr: {optimizer.param_groups[0]["lr"]: .3e}, '
                f"Accuracy: {accuracy}"
            )
            if accuracy > best_accuracy:
                logger.info(
                    "Best performance so far. best accuracy:{}".format(accuracy)
                )
                # model.save_pretrained(output_dir)
                # tokenizer.save_pretrained(output_dir)
                # torch.save(args, os.path.join(output_dir, "training_args.bin"))
                best_accuracy = accuracy
                best_dev_epoch = epoch

        if attack_setting.do_pgd_attack:
            do_pgd_attack(model, valid_loader, attack_setting)

        do_textattack_attack(
            model,
            tokenizer=tokenizer,
            dataset_setting=dataset_setting,
            attack_setting=attack_setting,
        )


# 不会修改模型,在attack之后不会保留梯度信息
def do_pgd_attack(model, dev_loader, attack_setting):
    model.eval()
    pbar = tqdm(dev_loader)
    correct = 0
    total = 0
    avg_loss = utils.ExponentialMovingAverage()
    for model_inputs, labels in pbar:
        word_embedding_layer = model.get_input_embeddings()
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        embedding_init = word_embedding_layer(input_ids)

        if attack_setting.adv_init_mag > 0:
            input_mask = attention_mask.to(embedding_init)
            input_lengths = torch.sum(input_mask, 1)
            if attack_setting.adv_norm_type == "l2":
                delta = torch.zeros_like(embedding_init).uniform_(
                    -1, 1
                ) * input_mask.unsqueeze(2)
                dims = input_lengths * embedding_init.size(-1)
                magnitude = attack_setting.adv_init_mag / torch.sqrt(dims)
                delta = delta * magnitude.view(-1, 1, 1)
            elif attack_setting.adv_norm_type == "linf":
                delta = torch.zeros_like(embedding_init).uniform_(
                    -attack_setting.adv_init_mag, attack_setting.adv_init_mag
                ) * input_mask.unsqueeze(2)
        else:
            delta = torch.zeros_like(embedding_init)

        total_loss = 0.0
        for astep in range(attack_setting.adv_steps):
            # (0) forward
            delta.requires_grad_()
            batch = {
                "inputs_embeds": delta + embedding_init,
                "attention_mask": attention_mask,
            }
            logits = model(**batch).logits
            preds = logits.argmax(dim=-1)
            # (1) backward
            losses = F.cross_entropy(logits, labels)
            loss = torch.mean(losses)
            # loss = loss / adv_steps
            total_loss += loss.item()
            model.zero_grad()
            loss.backward()
            # loss.backward(retain_graph=True)

            # (2) get gradient on delta
            delta_grad = delta.grad.clone().detach()

            # (3) update and clip
            if attack_setting.adv_norm_type == "l2":
                denorm = torch.norm(
                    delta_grad.view(delta_grad.size(0), -1), dim=1
                ).view(-1, 1, 1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + attack_setting.adv_lr * delta_grad / denorm).detach()
                if attack_setting.adv_max_norm > 0:
                    delta_norm = torch.norm(
                        delta.view(delta.size(0), -1).float(), p=2, dim=1
                    ).detach()
                    exceed_mask = (delta_norm > attack_setting.adv_max_norm).to(
                        embedding_init
                    )
                    reweights = (
                        attack_setting.adv_max_norm / delta_norm * exceed_mask
                        + (1 - exceed_mask)
                    ).view(-1, 1, 1)
                    delta = (delta * reweights).detach()
            elif attack_setting.adv_norm_type == "linf":
                denorm = torch.norm(
                    delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")
                ).view(-1, 1, 1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + attack_setting.adv_lr * delta_grad / denorm).detach()

        delta.requires_grad = False
        batch = {
            "inputs_embeds": delta + embedding_init,
            "attention_mask": attention_mask,
        }
        # optimizer.zero_grad()
        logits = model(**batch).logits

        losses = F.cross_entropy(logits, labels)
        loss = torch.mean(losses)
        avg_loss.update(loss.item())

        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    pgd_accuracy = correct / (total + 1e-13)
    pgd_aua = pgd_accuracy
    logger.info(f"PGD Aua: {pgd_accuracy}")
    logger.info(f"PGD Loss: {avg_loss.get_metric()}")

    model.train()
    model.zero_grad()
    return pgd_accuracy, avg_loss


def do_textattack_attack(model, tokenizer, dataset_setting, attack_setting):
    # attack_seed = 42
    model.eval()
    from attack import do_attack_with_model

    if attack_setting.attack_all:
        attack_methods = [
            "textfooler",
            "textbugger",
            "bertattack",
        ]  # todo done bertattack放最后一个，因为要改变攻击的参数！！！
    else:
        attack_methods = [attack_setting.attack_method]

    for attack_method in attack_methods:
        info = do_attack_with_model(
            model, tokenizer, dataset_setting, attack_setting, attack_method
        )
        logger.info(info)

    model.train()
    model.zero_grad()


if __name__ == "__main__":
    config = utils.read_config("train/soft_label.yaml")

    soft_label_train(config)
