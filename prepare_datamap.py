

from pathlib import Path
import numpy as np
from tqdm import tqdm

import settings
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    AdamW, AutoConfig, AutoTokenizer,get_linear_schedule_with_warmup 
)
import utils
import time
from loguru import logger
cur_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime(time.time() + 8 * 60 * 60))
file_name = Path(__file__).name.split(".")[0]
log_file = f"log/{file_name}/{cur_time}.log"
logger.add(log_file, rotation="100 MB",diagnose=True,backtrace=True,enqueue=True,retention="10 days",compression="zip")

import os
from torch import nn
# from transformers.models.bert.modeling_bert import BertSelfAttention, BertLayer # modified
from transformers.models.bert.modeling_bert import BertSelfAttention, BertLayer # modified
# from modeling_utils import PreTrainedModel
# 对抗训练的英文：Adversarial Training Methods for Semi-Supervised Text Classification

import datasets

# 对所有数据进行评估，并不训练
def robust_statistics(model,train_dev_loader_with_idx,adv_setting,use_cur_preds=True):
    config = adv_setting
    statistics={i.as_py():{} for i in train_dev_loader_with_idx.dataset.get_column("idx")}

    model.eval()
    pbar = tqdm(train_dev_loader_with_idx)
    for model_inputs, labels,idxs in pbar:
        logits = model(**model_inputs).logits
        preds = logits.argmax(dim=-1)
        for cur_logits, cur_label, cur_pred,idx in zip(logits, labels, preds,idxs):
            if use_cur_preds:
                cur_losses = F.cross_entropy(cur_logits.unsqueeze(0), cur_pred.unsqueeze(0))
            else:
                cur_losses = F.cross_entropy(cur_logits.unsqueeze(0),cur_label.unsqueeze(0))
            cur_loss = torch.mean(cur_losses)
            single_statistics = {
                "golden_label": cur_label.item(),
                "original_loss": cur_loss.item(),
                "original_pred": (cur_label.item()==cur_pred.item()),
                "original_logit": cur_logits[cur_label.item()].item(),
                "original_probability": nn.Softmax(dim=-1)(cur_logits)[cur_label.item()].item(),
            }
            statistics[idx].update(single_statistics)
        pbar.set_description("Doing original statistics")
        

    model.train()
    pbar = tqdm(train_dev_loader_with_idx)
    for model_inputs, labels,idxs in pbar:
        #获取当前batch的预测结果
        if use_cur_preds:
            cur_batch_preds = model(**model_inputs).logits.argmax(dim=-1)

        # 获取embedding_init
        word_embedding_layer = model.get_input_embeddings()
        input_ids = model_inputs['input_ids']
        attention_mask = model_inputs['attention_mask']
        embedding_init = word_embedding_layer(input_ids)
        # initialize delta
        if config.adv_init_mag > 0:
            input_mask = attention_mask.to(embedding_init)
            input_lengths = torch.sum(input_mask, 1)
            if config.adv_norm_type == 'l2':
                delta = torch.zeros_like(embedding_init).uniform_(-1, 1) * input_mask.unsqueeze(
                    2)
                dims = input_lengths * embedding_init.size(-1)
                magnitude = config.adv_init_mag / torch.sqrt(dims)
                delta = (delta * magnitude.view(-1, 1, 1))
            elif config.adv_norm_type == 'linf':
                delta = torch.zeros_like(embedding_init).uniform_(-config.adv_init_mag,
                                                                  config.adv_init_mag) * input_mask.unsqueeze(2)
        else:
            delta = torch.zeros_like(embedding_init)
        # freelb
        total_loss = 0.0
        model.zero_grad()
        for astep in range(config.adv_steps):
            # 0. forward
            delta.requires_grad_()
            batch = {'inputs_embeds': delta + embedding_init, 'attention_mask': attention_mask}
            logits = model(**batch).logits
            preds = logits.argmax(dim=-1)
            # 1.loss backward
            if use_cur_preds:
                losses = F.cross_entropy(logits, cur_batch_preds.squeeze(-1))
            else:
                losses = F.cross_entropy(logits, labels.squeeze(-1))
            loss = torch.mean(losses)
            loss = loss / config.adv_steps
            total_loss += loss.item()
            
            loss.backward()

            if astep == config.adv_steps - 1:
                for cur_logits, cur_label, cur_pred,cur_batch_pred, idx,cur_delta in zip(logits, labels, preds,cur_batch_preds,idxs,delta):
                    if use_cur_preds:
                        cur_losses = F.cross_entropy(cur_logits.unsqueeze(0), cur_batch_pred.unsqueeze(0))
                    else:
                        cur_losses = F.cross_entropy(cur_logits.unsqueeze(0), cur_label.unsqueeze(0))
                    cur_loss = torch.mean(cur_losses)
                    single_statistics = {
                        "after_perturb_loss": cur_loss.item(),
                        "after_perturb_pred": (cur_label.item() == cur_pred.item()),
                        "after_perturb_logit": cur_logits[cur_label.item()].item(),
                        "after_perturb_probability": nn.Softmax(dim=-1)(cur_logits)[cur_label.item()].item(),
                        "delta_norm": torch.norm(cur_delta, p=2, keepdim=False).item(),
                    }
                    statistics[idx].update(single_statistics)
                break
            # 2. get gradient on delta
            delta_grad = delta.grad.clone().detach()
            # 3. update and clip
            if config.adv_norm_type == "l2":
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + config.adv_lr * delta_grad / denorm).detach()  # detach 截断反向传播?
                if config.adv_max_norm > 0:
                    delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                    exceed_mask = (delta_norm > config.adv_max_norm).to(embedding_init)
                    reweights = (config.adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1,
                                                                                                        1)
                    # 将权重大于config.adv_max_norm的部分更新权重为config.adv_max_norm / delta_norm * exceed_mask
                    delta = (delta * reweights).detach()
            elif config.adv_norm_type == "linf":
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1,
                                                                                                         1, 1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + config.adv_lr * delta_grad / denorm).detach()
            embedding_init = word_embedding_layer(input_ids)  # 重新初始化embedding
        pbar.set_description("Doing perturbation statistics")
    return statistics


def robust_statistics_fgsm(model,train_dev_loader_with_idx,adv_setting,use_cur_preds=True):
    #rename
    config = adv_setting
    statistics={idx:{} for idx in train_dev_loader_with_idx.dataset.get_column("idx")}
    
    model.eval()
    pbar = tqdm(train_dev_loader_with_idx)
    for model_inputs, labels,idxs in pbar:
        logits = model(**model_inputs).logits
        preds = logits.argmax(dim=-1)
        for cur_logits, cur_label, cur_pred, idx in zip(logits, labels, preds,idxs):
            if use_cur_preds:
                cur_losses = F.cross_entropy(cur_logits.unsqueeze(0), cur_pred.unsqueeze(0))
            else:
                cur_losses = F.cross_entropy(cur_logits.unsqueeze(0),cur_label.unsqueeze(0))
            cur_loss = torch.mean(cur_losses)
            single_statistics = {
                "golden_label": cur_label.item(),
                "original_loss": cur_loss.item(),
                "original_pred": (cur_label.item()==cur_pred.item()),
                "original_logit": cur_logits[cur_label.item()].item(),
                "original_probability": nn.Softmax(dim=-1)(cur_logits)[cur_label.item()].item(),
            }
            statistics[idx].update(single_statistics)
        pbar.set_description("Doing original statistics")
        # pass

    model.train()
    pbar = tqdm(train_dev_loader_with_idx)
    for model_inputs, labels,idxs in pbar:
        if use_cur_preds:
            cur_batch_preds = model(**model_inputs).logits.argmax(dim=-1)
        # for fgsm,get embedding init
        word_embedding_layer = model.get_input_embeddings()
        input_ids = model_inputs['input_ids']
        attention_mask = model_inputs['attention_mask']
        embedding_init = word_embedding_layer(input_ids)
        # initialize delta
        delta = torch.zeros_like(embedding_init)
        total_loss = 0.0

        # 0. forward
        embedding_init.requires_grad_()
        delta.requires_grad_()
        batch = {'inputs_embeds': delta + embedding_init, 'attention_mask': attention_mask}
        logits = model(**batch).logits
        preds = logits.argmax(dim=-1)
        # 1.
        if use_cur_preds:
            losses = F.cross_entropy(logits, cur_batch_preds.squeeze(-1))
        else:
            losses = F.cross_entropy(logits, labels.squeeze(-1))
        # losses = F.cross_entropy(logits, labels.squeeze(-1))
        loss = torch.mean(losses)
        total_loss += loss.item()
        
        
        model.zero_grad()
        loss.backward()
        # 2. get gradient on delta
        delta_grad = delta.grad.clone().detach()
        # 3. update and clip
        if config.adv_norm_type == "l2":
            denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
            denorm = torch.clamp(denorm, min=1e-8)
            delta = (delta + config.adv_lr * delta_grad / denorm).detach()  # detach 截断反向传播?
            if config.adv_max_norm > 0:
                delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                exceed_mask = (delta_norm > config.adv_max_norm).to(embedding_init)
                reweights = (config.adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1,
                                                                                                    1)
                delta = (delta * reweights).detach()
        elif config.adv_norm_type == "linf":
            denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1,
                                                                                                     1, 1)
            denorm = torch.clamp(denorm, min=1e-8)
            delta = (delta + config.adv_lr * delta_grad / denorm).detach()
        else :
            raise NotImplementedError

        # 4. forward
        delta.requires_grad_()
        batch = {'inputs_embeds': delta + embedding_init, 'attention_mask': attention_mask}
        logits = model(**batch).logits
        preds = logits.argmax(dim=-1)
        for cur_logits, cur_label, cur_pred,cur_delta,cur_batch_pred, idx in zip(logits, labels, preds,delta,cur_batch_preds,idxs):
            if use_cur_preds:
                cur_losses = F.cross_entropy(cur_logits.unsqueeze(0), cur_batch_pred.unsqueeze(0))
            else:
                cur_losses = F.cross_entropy(cur_logits.unsqueeze(0), cur_label.unsqueeze(0))
            cur_loss = torch.mean(cur_losses)
            single_statistics = {
                "after_perturb_loss": cur_loss.item(),
                "after_perturb_pred": (cur_label.item() == cur_pred.item()),
                "after_perturb_logit": cur_logits[cur_label.item()].item(),
                "after_perturb_probability": nn.Softmax(dim=-1)(cur_logits)[cur_label.item()].item(),
                "delta_norm": torch.norm(cur_delta, p=2, keepdim=False).item(),
            }
            statistics[idx].update(single_statistics)
        pbar.set_description("Doing perturbation statistics")

    return statistics

# fine tuning 模型，并统计微调过程中的统计量
@logger.catch
def prepare_datamap(config):
    settings.set_seed(config.seed)
    logger.info("prepare datamap start")
    logger.info(config)
    dataset_setting = config.dataset_setting
    train_setting = config.train_setting
    model_setting = config.model_setting
    statistics_setting = config.statistics_setting
    adverisal_setting = config.adverisal_setting
    
    model,tokenizer = settings.get_model_tokenizer(model_setting,dataset_setting.num_labels)
    train_loader = settings.get_dataloader(tokenizer,dataset_setting,'train',with_idx = True)
    optimizer = settings.get_optimizer(model,train_setting)
    scheduler = settings.get_scheduler(optimizer,train_setting,train_loader)

    valid_loader = settings.get_dataloader(tokenizer,dataset_setting,'validation')
    
    robust_statistics_dict = {}

    best_accuracy = 0
    global_step = 0
    epoch_steps = len(train_loader.dataset) // train_setting.batch_size
    for epoch in range(train_setting.epochs):
        
        avg_loss = utils.ExponentialMovingAverage()
        model.train()
        pbar = tqdm(train_loader)
        for model_inputs, labels,_ in pbar:
            if (global_step % int(statistics_setting.interval * epoch_steps) == 0 
                    and(statistics_setting.with_untrained_model or global_step != 0)):
                statistics_func = robust_statistics_fgsm if statistics_setting.use_fgsm else robust_statistics
                cur_robust_statistics = statistics_func(model, train_loader,adverisal_setting,statistics_setting.use_cur_preds)
                robust_statistics_dict[global_step] = cur_robust_statistics
            # forward
            logits = model(**model_inputs).logits
            preds = logits.argmax(dim=-1)
            losses = F.cross_entropy(logits,labels)
            loss = torch.mean(losses)
            # backward
            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            # 更新统计量
            avg_loss.update(loss.item())
            pbar.set_description(f'epoch: {epoch: d}, '
                                    f'loss: {avg_loss.get_metric(): 0.4f}, '
                                    f'lr: {optimizer.param_groups[0]["lr"]: .3e}')
            global_step+=1
        

        if train_setting.do_eval:
            logger.info('Evaluating...')
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for model_inputs, labels in valid_loader:
                    model_inputs = {k: v.to(settings.device) for k, v in model_inputs.items()}
                    labels = labels.to(settings.device)
                    logits = model(**model_inputs,return_dict=False)[0]
                    preds = logits.argmax(dim=-1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                accuracy = correct / (total + 1e-13)
            logger.info(f'Epoch: {epoch}, '
                        f'Loss: {avg_loss.get_metric(): 0.4f}, '
                        f'Lr: {optimizer.param_groups[0]["lr"]: .3e}, '
                        f'Accuracy: {accuracy}')

            if accuracy > best_accuracy:
                logger.info(f'Best performance so far.accracy: {accuracy}, epoch: {epoch}')
                best_accuracy = accuracy

    # 保存robust_statistics_dict 和 config，文件命名为时间，方便查找
    if os.path.exists("statistics/npys/") is False:
        os.mkdir("statistics/npys/")
    # 保存统计量，但是config另外存储
    npy_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime(time.time() + 8 * 60 * 60))
    npy_file = f'statistics/npys/{npy_time}.npy'
    np.save(npy_file,{})
    config['log_file'] = log_file
    config['npy_file'] = npy_file
    utils.write_config(config,"statistics/meta.yaml")
    
    logger.info("prepare datamap end")
    

if __name__ == '__main__':
    yaml_file = "statistics/data_map.yaml"
    configs = utils.read_configs(yaml_file)
    for config in configs:
        prepare_datamap(config)
