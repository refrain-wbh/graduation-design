import argparse
import logging
import os
from pathlib import Path
import random
import numpy as np
from tqdm import tqdm
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    AdamW, AutoConfig, AutoTokenizer
)
from models.modeliing_bert import BertForSequenceClassification
from models.modeling_roberta import RobertaForSequenceClassification
from torch.utils.tensorboard import SummaryWriter

import utils,settings,select_data
# from transformers.models.bert.modeling_bert import BertSelfAttention, BertLayer # modified
from transformers.models.bert.modeling_bert import BertSelfAttention, BertLayer # modified

logger = settings.get_logger(__file__)

def evaluate(model,dev_loader):
    logger.info('Evaluating...')
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
    logger.info(f'Clean Aua: {accuracy}')
    logger.info(f'Clean Loss: {avg_loss.get_metric()}')
    return accuracy,avg_loss
    # logger.info(f'Best dev metric: {best_accuracy} in Epoch: {best_dev_epoch}')


data_indices_to_new_id={}
new_id_to_data_indices={}

@logger.catch
def finetune(args):
    tensorboard_path = "/root/workspace/Robust-Data/runs/new_finetune_soft_label_from_init_not_symmetry"+args.output_dir[args.output_dir.rfind("/"):]
    writer = SummaryWriter(tensorboard_path)
    
    
    settings.set_seed(config.seed)
    
    
    dataset_setting = config.dataset_setting
    train_setting = config.train_setting
    model_setting = config.model_setting
    statistics_setting = config.statistics_setting
    adverisal_setting = config.adverisal_setting
    
    model,tokenizer = settings.get_model_tokenizer(model_setting,dataset_setting.num_labels)
    optimizer = settings.get_optimizer(model,train_setting)
    scheduler = settings.get_scheduler(optimizer,train_setting,train_loader)
    
    train_loader = settings.get_dataloader(tokenizer,dataset_setting,"train",with_idx = True)
    valid_loader = settings.get_dataloader(tokenizer,dataset_setting,"valid")
    if train_setting.do_test:
        test_loader = settings.get_dataloader(tokenizer,dataset_setting,"test")
    df = select_data.get_df_from_statistics_file(statistics_setting.statistics_file)
    result_data_indices, selected_label_nums = select_data.generate_data_indices(statistics_setting,df,dataset_setting.num_labels)
    logger.debug(f"selected_label_nums:{selected_label_nums}")

    if args.show_data > 0:
        import csv
        print("Metric:{}".format(args.select_metric))
        for i in range(args.show_data):
            print("sentence: {}, label: {}".format(train_dataset.dataset["sentence"][i],train_dataset.dataset["label"][i]))
            print()
            print()
            show_data_dir = "/root/workspace/Robust-Data/analysis_experiments/show_data/"
            show_data_file = "/root/workspace/Robust-Data/analysis_experiments/show_data/show_data.csv"
            show_data_format = [
               "select_metric","seed", "sentence","label","order_in_cur_metric"
            ]
            if not os.path.exists(show_data_file):
                # os.makedirs(show_data_dir)
                out_csv = open(show_data_file, 'a', encoding='utf-8', newline="")
                csv_writer = csv.writer(out_csv)
                cur_row = [i for i in show_data_format]
                csv_writer.writerow(cur_row)
                out_csv.close()

            # 写入数据
            out_csv = open(show_data_file, 'a', encoding='utf-8', newline="")
            csv_writer = csv.writer(out_csv)
            cur_row = []
            cur_row.append(args.select_metric)
            cur_row.append(args.seed)
            cur_row.append(train_dataset.dataset["sentence"][i])
            cur_row.append(train_dataset.dataset["label"][i])
            cur_row.append(i)
            csv_writer.writerow(cur_row)
            out_csv.close()

        return

    if args.do_pgd_training:
        log_format = ["statistics_source",
                      "select_metric",
                      "select_ratio", "ratio_proportion",
                      "selected_label_nums",
                      "lr",
                      "seed", "epochs",
                      "pgd_step", "pgd_lr",
                      "clean", "pgd_aua", "attack_method","beta","soft_label","pgd_adv_steps","pgd_adv_steps2","pgd_adv_lr","pgd_adv_lr2"
                      ]
    else:
        log_format = ["statistics_source",
                      "select_metric",
                      "select_ratio", "ratio_proportion",
                      "selected_label_nums",
                      "lr",
                      "seed", "epochs",
                      "pgd_step", "pgd_lr",
                      "clean", "pgd_aua", "attack_method","beta","soft_label"
                      ]

    
    one_epoch_steps = int(len(train_dataset) // args.bsz)
    if args.save_steps<1 and args.save_steps>0:
        args.save_steps = int(one_epoch_steps*args.save_steps)
    save_steps = args.save_steps

    best_accuracy = 0
    global_step = 0
    
    for epoch in range(args.epochs):
        avg_loss = utils.ExponentialMovingAverage()
        model.train()
        pbar = tqdm(train_loader, desc="Epoch {}".format(epoch))
        for model_inputs, labels, indices in pbar:
            selected_set = set(result_data_indices)
            data_selected  = torch.tensor([1 if idx in selected_set  else 0 for idx in indices])
            data_not_selected = torch.tensor([0 if idx in selected_set  else 1 for idx in indices])
            if save_steps > 0 and global_step % save_steps ==0:
                save_model_one_step(args,global_step,model,output_dir=args.output_dir,tokenizer=tokenizer)
            batch_loss = 0
            logits = model(**model_inputs).logits
            preds = logits.argmax(dim=-1)
            
            losses_hard = F.cross_entropy(logits, labels.squeeze(-1),reduction="none")
            losses_soft = SoftCrossEntropy(logits,labels.squeeze(-1),reduction="none",
                                           soft_label=train_setting.soft_label,device=settings.device)

            
            soft_coef = train_setting.beta * (torch.ones_like(losses_soft))
            hard_loss = data_not_selected.to(settings.device).mul(losses_hard)
            soft_loss = data_selected.to(settings.device).mul(soft_coef.to(settings.device)).mul(losses_soft)
            losses = soft_loss+hard_loss
            loss = torch.mean(losses)

            model.zero_grad()
            loss.backward()
            
            # 梯度裁剪，避免出现梯度爆炸情况
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            avg_loss.update(batch_loss)

            writer.add_scalars("train_loss", {"whole_loss":avg_loss.get_metric()},global_step)

            # args.writer.add_scalars("train_loss/whole_loss", avg_loss.get_metric(), global_step=global_step, walltime=None)

            pbar.set_description(f'epoch: {epoch: d}, '
                                f'loss: {avg_loss.get_metric(): 0.4f}, '
                                f'lr: {optimizer.param_groups[0]["lr"]: .3e},'
                                # f'non_robust_soft:{non_robust_soft_loss_avg_all.get_metric():0.4f},'
                                # f'non_robust_hard:{non_robust_hard_loss_avg_all.get_metric():0.4f},'
                                # f'robust_soft:{robust_soft_loss_avg_all.get_metric():0.4f},'
                                # f'robust_hard:{robust_hard_loss_avg_all.get_metric():0.4f},'
                                )
            global_step+=1
        #valid数据集上验证
        if args.do_eval and not args.cal_time:
            accuracy,clean_loss = evaluate(model, valid_loader)
            logger.info(f'Epoch: {epoch}, '
                        f'Loss: {avg_loss.get_metric(): 0.4f}, '
                        f'Lr: {optimizer.param_groups[0]["lr"]: .3e}, '
                        f'Accuracy: {accuracy}')
            if accuracy > best_accuracy:
                logger.info('Best performance so far. best accuracy:{}'.format(accuracy))
                # model.save_pretrained(output_dir)
                # tokenizer.save_pretrained(output_dir)
                # torch.save(args, os.path.join(output_dir, "training_args.bin"))
                best_accuracy = accuracy
                best_dev_epoch = epoch
    
    # train over and do text or pgd attack
    if args.attack_every_step > 0:
        for step in range(num_training_steps):
            if step%args.save_steps==0:

                one_epoch_steps = int(len(train_dataset) // args.bsz)

                epoch = int(step//one_epoch_steps)
                logger.info("current step:{},current epoch:{}".format(step, epoch))
                args.cur_epoch = epoch
                args.cur_step = step
                s = Path(str(output_dir) + '/step' + str(step))
                model = settings.get_model_base(train_setting.model_name)
                config = AutoConfig.from_pretrained(train_setting.model_name, num_labels=dataset_settings.num_labels,mirror='tuna')
                model.from_pretrained(s,config = config)
                
                if args.do_pgd_attack:
                    pgd_aua,pgd_loss = do_pgd_attack( dev_loader, device, model,
                                                adv_steps=args.pgd_step,
                                                adv_lr=args.pgd_lr,
                                                adv_init_mag=args.adv_init_mag,
                                                adv_max_norm=args.adv_max_norm,
                                                adv_norm_type=args.adv_norm_type
                                                )
                    optimizer.zero_grad()
                    args.pgd_aua = pgd_aua
                    args.pgd_loss = pgd_loss.get_metric()
                else:
                    args.pgd_aua = 0
                    args.pgd_loss = 0


                do_textattack_attack(args, model, tokenizer,
                                        do_attack=args.do_attack,
                                        attack_seed=42,
                                        attack_all=args.attack_all,
                                        attack_method="textfooler",
                                        attack_every_epoch=False,
                                        attack_every_step=True,
                                        log_format=[i for i in log_format]

                                        )

    elif args.attack_every_epoch>0:
        for epoch in range(args.epochs):
            logger.info("current epoch:{}".format( epoch))
            if args.cycle_train > 0:
                if epoch%(args.cycle_train * 2) < args.cycle_train: #
                    logger.info("current metric:{},cureent ratio:{}".format(args.select_metric,args.select_ratio))
                    cur_select_metric = args.select_metric
                    args.cur_select_metric = cur_select_metric

                else:
                    logger.info("current metric:{},cureent ratio:{}".format(args.select_metric2,args.select_ratio2))
                    cur_select_metric = args.select_metric2
                    args.cur_select_metric = cur_select_metric
            else:
                logger.info("current metric:{},cureent ratio:{}".format(args.select_metric, args.select_ratio))
                cur_select_metric = args.select_metric
                args.cur_select_metric = cur_select_metric

            args.cur_epoch = epoch
            s = Path(str(output_dir) + '/epoch' + str(epoch))
            if args.model_name == "bert-base-uncased":
                model = BertForSequenceClassification.from_pretrained(s, config=config)
            elif args.model_name == "roberta-base":
                model = RobertaForSequenceClassification.from_pretrained(s, config=config)
            else:
                model = BertForSequenceClassification.from_pretrained(s, config=config)
            model.to(device)
            if args.do_eval and not args.cal_time:
                if args.attack_every_epoch ==10:
                    args.attack_dataset_metric = args.select_metric
                    loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)
                elif args.attack_every_epoch ==20:
                    args.attack_dataset_metric = args.select_metric2
                    loader = DataLoader(train_dataset2, batch_size=args.bsz, shuffle=True, collate_fn=collator)
                else:
                    args.attack_dataset_metric = "test_set"
                    loader = dev_loader
                clean,clean_loss = evaluate(loader, device, model)
                args.clean = clean
                args.clean_loss = clean_loss.get_metric()
            else:
                args.clean = 0
                args.clean_loss = 0
            if args.do_pgd_attack:
                if args.attack_every_epoch ==10:
                    args.attack_dataset_metric = args.select_metric
                    loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)
                elif args.attack_every_epoch ==20:
                    args.attack_dataset_metric = args.select_metric2
                    loader = DataLoader(train_dataset2, batch_size=args.bsz, shuffle=True, collate_fn=collator)
                else:
                    args.attack_dataset_metric = "test_set"
                    loader = dev_loader
                pgd_aua,pgd_loss = do_pgd_attack( loader, device, model,
                                            adv_steps=args.pgd_step,
                                            adv_lr=args.pgd_lr,
                                            adv_init_mag=args.adv_init_mag,
                                            adv_max_norm=args.adv_max_norm,
                                            adv_norm_type=args.adv_norm_type
                                            )
                optimizer.zero_grad()
                args.pgd_aua = pgd_aua
                args.pgd_loss = pgd_loss.get_metric()
            else:
                args.pgd_aua = 0
                args.pgd_loss = 0

            do_textattack_attack(args, model, tokenizer,
                                    do_attack=args.do_attack,
                                    attack_seed=42,
                                    attack_all=args.attack_all,
                                    attack_method="textfooler",
                                    attack_every_epoch=True,
                                    attack_every_step=False,
                                    log_format=[i for i in log_format]
                                    )

    if (not args.attack_every_epoch and not args.attack_every_step and args.do_attack):
        if args.do_eval and not args.cal_time:
            args.attack_dataset_metric = "test_set"
            loader = dev_loader
            clean, clean_loss = evaluate(loader, device, model)
            args.clean = clean
            args.clean_loss = clean_loss
        else:
            args.clean = 0
            args.clean_loss = 0
        if args.do_pgd_attack:
            pgd_aua,pgd_loss = do_pgd_attack( dev_loader, device, model,
                                                adv_steps=args.pgd_step,
                                                adv_lr=args.pgd_lr,
                                                adv_init_mag=args.adv_init_mag,
                                                adv_max_norm=args.adv_max_norm,
                                                adv_norm_type=args.adv_norm_type
                                                )
            args.pgd_aua = pgd_aua
            args.pgd_loss = pgd_loss
        else:
            args.pgd_aua = 0
            args.pgd_loss = 0
        optimizer.zero_grad()
        do_textattack_attack(args, model, tokenizer,
                                do_attack=args.do_attack,
                                attack_seed=42,
                                attack_all=args.attack_all,
                                attack_method="textfooler",
                                attack_every_epoch=args.attack_every_epoch,
                                log_format=[i for i in log_format]
                                )
    elif (not args.attack_every_epoch and not args.attack_every_step and not args.do_attack and args.do_pgd_attack):
        pgd_aua,pgd_loss = do_pgd_attack( dev_loader, device, model,
                                            adv_steps=args.pgd_step,
                                            adv_lr=args.pgd_lr,
                                            adv_init_mag=args.adv_init_mag,
                                            adv_max_norm=args.adv_max_norm,
                                            adv_norm_type=args.adv_norm_type,
                                            )
        optimizer.zero_grad()
        args.pgd_aua = pgd_aua
        args.pgd_loss = pgd_loss



def calculate_perturbed_loss_one_batch(args, model, model_inputs, labels,adv_steps,adv_lr):
    model.eval()
    # perturbed loss计算
    word_embedding_layer = model.get_input_embeddings()
    input_ids = model_inputs['input_ids']
    attention_mask = model_inputs['attention_mask']
    embedding_init = word_embedding_layer(input_ids)
    correct = 0
    total = 0
    # if args.adv_init_mag > 0:
    #     input_mask = attention_mask.to(embedding_init)  # 对embedding做mask？
    #     input_lengths = torch.sum(input_mask, 1)
        # if args.adv_norm_type == 'l2':
        #     delta = torch.zeros_like(embedding_init).uniform_(-1, 1) * input_mask.unsqueeze(
        #         2)  # 有些会被mask掉，所以乘以* input_mask.unsqueeze(2)
        #     dims = input_lengths * embedding_init.size(-1)
        #     magnitude = args.adv_init_mag / torch.sqrt(dims)
        #     delta = (delta * magnitude.view(-1, 1, 1))
        # elif args.adv_norm_type == 'linf':
        # delta = torch.zeros_like(embedding_init).uniform_(-args.adv_init_mag,
        #                                                       args.adv_init_mag) * input_mask.unsqueeze(2)
    # else:
    delta = torch.zeros_like(embedding_init)
    for astep in range(adv_steps):
        # 0. forward
        delta.requires_grad_()
        batch = {'inputs_embeds': delta + embedding_init, 'attention_mask': attention_mask}
        logits = model(**batch, return_dict=False)[0]
        _, preds = logits.max(dim=-1)
        # 1.
        losses = F.cross_entropy(logits, labels.squeeze(-1))
        # losses = F.cross_entropy(logits, labels.squeeze(-1))
        loss = torch.mean(losses)
        loss = loss / adv_steps
        loss.backward()

        if astep == adv_steps - 1:
            # todo 计算loss
            losses = F.cross_entropy(logits, labels.squeeze(-1),reduction="none")
            _, preds = logits.max(dim=-1)
            correct += (preds == labels.squeeze(-1)).sum().item()
            total += labels.size(0)
            model.train()
            model.zero_grad()
            embedding_init = word_embedding_layer(input_ids)
            delta.requires_grad = False
            return losses,correct,total
            # pass

        # 2. get gradient on delta
        delta_grad = delta.grad.clone().detach()
        # 3. update and clip
        if args.adv_norm_type == "l2":
            denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
            denorm = torch.clamp(denorm, min=1e-8)
            delta = (delta + adv_lr * delta_grad / denorm).detach()  # detach 截断反向传播?
            if args.adv_max_norm > 0:
                delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                exceed_mask = (delta_norm > args.adv_max_norm).to(embedding_init)
                reweights = (args.adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1,
                                                                                                    1)
                delta = (delta * reweights).detach()
        elif args.adv_norm_type == "linf":
            denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1,
                                                                                                     1, 1)
            denorm = torch.clamp(denorm, min=1e-8)
            delta = (delta + adv_lr * delta_grad / denorm).detach()
        model.zero_grad()
        embedding_init = word_embedding_layer(input_ids)  # 重新初始化embedding



def train_one_epoch_new(args, avg_loss, device, epoch, model, optimizer, scheduler, train_loader, global_step, save_steps, tokenizer,result_data_indices,beta=1,writer=None,dev_loader=None):


    non_robust_soft_loss_avg_all = utils.ExponentialMovingAverage()
    non_robust_hard_loss_avg_all = utils.ExponentialMovingAverage()
    robust_hard_loss_avg_all = utils.ExponentialMovingAverage()
    robust_soft_loss_avg_all = utils.ExponentialMovingAverage()
    perturbed_robust_avg_all = utils.ExponentialMovingAverage()
    perturbed_non_robust_avg_all = utils.ExponentialMovingAverage()
    perturbed_whole_avg_all = utils.ExponentialMovingAverage()



    pbar = tqdm(train_loader,ncols=100)
    for model_inputs, labels,indices in pbar:
        selected_set = set(result_data_indices)
        data_selected  = torch.tensor([1 if int(i) in selected_set  else 0 for i in indices])
        data_not_selected = torch.tensor([0 if int(i) in selected_set  else 1 for i in indices])
        
        if save_steps > 0 and global_step % save_steps ==0:
            save_model_one_step(args,global_step,model,output_dir=args.output_dir,tokenizer=tokenizer)
        batch_loss = 0
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        labels = labels.to(device)
        model.zero_grad()
        logits = model(**model_inputs, return_dict=False)[0]
        _, preds = logits.max(dim=-1)

        losses_hard = F.cross_entropy(logits, labels.squeeze(-1),reduction="none")
        losses_soft = SoftCrossEntropy(logits,labels.squeeze(-1),reduction="none",soft_label=args.soft_label,device=device)
        probs = F.softmax(logits,dim=1)

        # non_robust_coef = beta * (1 - torch.tensor([probs[i][labels[i]] for i in range(len(labels))]))
        # non_robust_coef = beta * (torch.tensor([1 for i in range(len(labels))]))
        # 选择的数据是计算hard loss的，否则是计算soft loss的
        # robust_loss = data_selected.to(device).mul(losses_hard)
        # if args.non_robust_type=="soft":
        #     non_robust_loss = data_not_selected.to(device).mul(non_robust_coef.to(device)).mul(losses_soft)
        # else:
        #     non_robust_loss = data_not_selected.to(device).mul(non_robust_coef.to(device)).mul(losses_hard)

        # todo  我现在想让选择的数据做soft loss，没选择的做hard loss，怎么办？
        soft_coef = beta * (torch.tensor([1 for i in range(len(labels))]))
        hard_loss = data_not_selected.to(device).mul(losses_hard)
        soft_loss = data_selected.to(device).mul(soft_coef.to(device)).mul(losses_soft)
        losses = soft_loss+hard_loss
        # just test
        # perturbed_losses,correct,total = calculate_perturbed_loss_one_batch(args=args, model=model, model_inputs=model_inputs, labels=labels,adv_steps=args.pgd_step,adv_lr=args.pgd_lr)
        # model.zero_grad()
        #
        # if sum(data_not_selected) > 0:
        #     non_robust_soft_loss = data_not_selected.to(device).mul(losses_soft)
        #     non_robust_hard_loss = data_not_selected.to(device).mul(losses_hard)
        #     non_robust_soft_loss_sum = sum(non_robust_soft_loss)
        #     non_robust_hard_loss_sum = sum(non_robust_hard_loss)
        #     non_robust_soft_loss_avg_all.update(non_robust_soft_loss_sum*beta,sum(data_not_selected))
        #     non_robust_hard_loss_avg_all.update(non_robust_hard_loss_sum*beta,sum(data_not_selected))
        #
        #     perturbed_non_robust_losses = data_not_selected.to(device).mul(perturbed_losses)
        #     perturbed_non_robust_losses_sum = sum(perturbed_non_robust_losses).item()
        #     perturbed_non_robust_avg_all.update(perturbed_non_robust_losses_sum,sum(data_not_selected))
        # if sum(data_selected) > 0:
        #     robust_soft_loss = data_selected.to(device).mul(losses_soft)
        #     robust_hard_loss = data_selected.to(device).mul(losses_hard)
        #     robust_soft_loss_sum = sum(robust_soft_loss)
        #     robust_hard_loss_sum = sum(robust_hard_loss)
        #     robust_soft_loss_avg_all.update(robust_soft_loss_sum,sum(data_selected))
        #     robust_hard_loss_avg_all.update(robust_hard_loss_sum,sum(data_selected))
        #
        #     perturbed_robust_losses = data_selected.to(device).mul(perturbed_losses)
        #     perturbed_robust_losses_sum = sum(perturbed_robust_losses).item()
        #     perturbed_robust_avg_all.update(perturbed_robust_losses_sum,sum(data_selected))
        #
        # perturbed_whole_losses_sum = sum(perturbed_losses).item()
        # perturbed_whole_avg_all.update(perturbed_whole_losses_sum,len(labels))
        #
        # import math
        #
        # expected_soft_loss = - (args.soft_label * math.log(args.soft_label) + (1-args.soft_label) * math.log(1-args.soft_label))
        # writer.add_scalars("train_loss", {"expected_soft_loss":expected_soft_loss},global_step)
        # writer.add_scalars("train_loss", {"non_robust_soft_loss_avg_all":non_robust_soft_loss_avg_all.get_metric()},global_step)
        # writer.add_scalars("train_loss", {"non_robust_hard_loss_avg_all":non_robust_hard_loss_avg_all.get_metric()},global_step)
        # writer.add_scalars("train_loss", {"robust_soft_loss_avg_all":robust_soft_loss_avg_all.get_metric()},global_step)
        # writer.add_scalars("train_loss", {"robust_hard_loss_avg_all":robust_hard_loss_avg_all.get_metric()},global_step)
        # # args.writer.add_scalars("train_loss/non_robust_hard_loss_avg_all", non_robust_hard_loss_avg_all.get_metric(), global_step=global_step, walltime=None)
        # # args.writer.add_scalars("train_loss/robust_soft_loss_avg_all", robust_hard_loss_avg_all.get_metric(), global_step=global_step, walltime=None)
        # # args.writer.add_scalars("train_loss/robust_hard_loss_avg_all", robust_hard_loss_avg_all.get_metric(), global_step=global_step, walltime=None)
        #
        # writer.add_scalars("train_loss", {"perturbed_robust_avg_all":perturbed_robust_avg_all.get_metric()},global_step)
        # writer.add_scalars("train_loss", {"perturbed_non_robust_avg_all":perturbed_non_robust_avg_all.get_metric()},global_step)
        # writer.add_scalars("train_loss", {"perturbed_whole_avg_all":perturbed_whole_avg_all.get_metric()},global_step)
        #
        #
        # # 每50步，在test上面做一下
        # if global_step % 50 ==0:
        #     test_clean_acc,test_clean_loss = evaluate(dev_loader, device, model)
        #     test_perturbed_aua, test_pgd_loss = calculate_dev_set_perturbed_loss(args, dev_loader, device, model)
        #     writer.add_scalars("train_loss", {"test_clean_loss": test_clean_loss.get_metric()},
        #                        global_step)
        #
        #     writer.add_scalars("train_loss", {"test_perturbed_loss": test_pgd_loss.get_metric()},
        #                        global_step)
        #
        #     writer.add_scalars("train_acc", {"test_clean_acc": test_clean_acc},
        #                        global_step)
        #     writer.add_scalars("train_acc", {"test_perturbed_aua": test_perturbed_aua},
        #                        global_step)
        #
        # model.zero_grad()

        # continue
        # losses = robust_loss+non_robust_loss
        loss = torch.mean(losses)
        # loss2  = model(**model_inputs,return_dict=False)
        batch_loss = loss.item()



        loss.backward()
        # 梯度裁剪，避免出现梯度爆炸情况
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        model.zero_grad()
        avg_loss.update(batch_loss)

        writer.add_scalars("train_loss", {"whole_loss":avg_loss.get_metric()},global_step)

        # args.writer.add_scalars("train_loss/whole_loss", avg_loss.get_metric(), global_step=global_step, walltime=None)

        pbar.set_description(f'epoch: {epoch: d}, '
                             f'loss: {avg_loss.get_metric(): 0.4f}, '
                             f'lr: {optimizer.param_groups[0]["lr"]: .3e},'
                             # f'non_robust_soft:{non_robust_soft_loss_avg_all.get_metric():0.4f},'
                             # f'non_robust_hard:{non_robust_hard_loss_avg_all.get_metric():0.4f},'
                             # f'robust_soft:{robust_soft_loss_avg_all.get_metric():0.4f},'
                             # f'robust_hard:{robust_hard_loss_avg_all.get_metric():0.4f},'
                             )
        global_step+=1

    return global_step


def calculate_dev_set_perturbed_loss(args, dev_loader, device, model):
    model.eval()
    pbar = tqdm(dev_loader)
    correct = 0
    total = 0
    avg_loss = utils.ExponentialMovingAverage()
    for model_inputs, labels in pbar:
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        labels = labels.to(device)
        model.zero_grad()
        perturbed_losses,cur_correct,cur_total = calculate_perturbed_loss_one_batch(args=args, model=model, model_inputs=model_inputs,
                                                              labels=labels,adv_steps=args.pgd_step,adv_lr=args.pgd_lr)
        batch_loss = torch.mean(perturbed_losses).item()
        avg_loss.update(batch_loss)
        correct+=cur_correct
        total+=cur_total
    pgd_accuracy = correct / (total + 1e-13)
    pgd_aua = pgd_accuracy
    logger.info(f'test perturbed Aua: {pgd_accuracy}')
    logger.info(f'test perturbed Loss: {avg_loss.get_metric()}')
    model.train()
    model.zero_grad()
    return pgd_aua,avg_loss


def SoftCrossEntropy(inputs, target, reduction='none',soft_label=1,device="cuda"):
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

    new_target = F.one_hot(target,num_labels).to(device)
    # 非golden位置为0
    inverse_target = (torch.ones(inputs.shape).to(device) - new_target).to(device)

    new_target = new_target * soft_label + inverse_target * ((1-soft_label) / (num_labels-1))
    losses = torch.sum(torch.mul(log_likelihood, new_target),dim=1)
    if reduction == 'average':
        losses = torch.sum(losses) / batch
    elif reduction == "none":
        return losses
    elif reduction=="sum":
        losses = torch.sum(losses)

    return losses


if __name__ == '__main__':
    config = utils.get_config()
    settings.set_seed(config.seed)

    finetune(config)
