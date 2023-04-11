

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
config = utils.read_config('config/config.yaml')
from loguru import logger
logger.add("log/{name}.log", rotation="100 MB",diagnose=True,backtrace=True,enqueue=True,retention="10 days",compression="zip")

import os
from torch import nn
# from transformers.models.bert.modeling_bert import BertSelfAttention, BertLayer # modified
from transformers.models.bert.modeling_bert import BertSelfAttention, BertLayer # modified
# from modeling_utils import PreTrainedModel


def process_npy(statistics_path,len_dataset,dataset_name="glue",task_name="sst2",only_original_pred=True,
                use_normed_loss = False, use_delta_grad=False):
    """
    处理原始的npy数据，转化为以数据点号为索引的 new_data_loss_diff,new_data_original_correctness,new_data_flip_times
    :param statistics_path:
    :param len_dataset:
    :param dataset_name:
    :param task_name:
    :param only_original_pred:
    :return:
    """
    # 导入的npy是按照interval来组织的，不是按照每一个datapoint，处理成按照datapoint。
    test_npy = np.load(statistics_path, allow_pickle=True).item()
    intervals = list(test_npy)
    len_interval = len(test_npy)
    # len_dataset,_ = dataset_to_length_and_batch_size(dataset_name,task_name)
    new_data_loss_diff = [[] for i in range(len_dataset)]
    new_data_original_correctness = [0 for i in range(len_dataset)]
    new_data_flip_times = [0 for i in range(len_dataset)]
    new_data_original_loss = [[] for i in range(len_dataset)]
    new_data_perturbed_loss = [[] for i in range(len_dataset)]

    new_data_original_logit = [[] for i in range(len_dataset)]
    new_data_perturbed_logit = [[] for i in range(len_dataset)]
    new_data_logit_diff = [[] for i in range(len_dataset)]

    new_data_original_probability = [[] for i in range(len_dataset)]
    new_data_perturbed_probability = [[] for i in range(len_dataset)]
    new_data_probability_diff = [[] for i in range(len_dataset)]

    new_data_golden_label = [0 for i in range(len_dataset)]

    new_data_delta_grad = None
    if use_delta_grad:
        new_data_delta_grad = [[] for i in range(len_dataset)]
    tmp_idx = 0
    for interval in intervals:
        for data_idx in range(len_dataset):
            cur_data = test_npy[interval][data_idx]
            if only_original_pred:
                if cur_data["original_pred"]: # 预测正确的才会参与最终绘图
                    new_data_original_correctness[data_idx]+=1
                    new_data_original_loss[data_idx].append(cur_data["original_loss"])
                    new_data_perturbed_loss[data_idx].append(cur_data["after_perturb_loss"])
                    if cur_data.__contains__("original_logit"):
                        new_data_original_logit[data_idx].append(cur_data["original_logit"])
                        new_data_perturbed_logit[data_idx].append(cur_data["after_perturb_logit"])
                        new_data_logit_diff[data_idx].append(cur_data["logit_diff"])

                        new_data_original_probability[data_idx].append(cur_data["original_probability"])
                        new_data_perturbed_probability[data_idx].append(cur_data["after_perturb_probability"])
                        new_data_probability_diff[data_idx].append(cur_data["probability_diff"])

                        new_data_golden_label[data_idx] = cur_data["golden_label"]

                    if not use_normed_loss:
                        new_data_loss_diff[data_idx].append(cur_data["loss_diff"])
                    else:
                        new_data_loss_diff[data_idx].append(cur_data["normed_loss_diff"])
                    if use_delta_grad:
                        new_data_delta_grad[data_idx].append(cur_data["delta_grad"])

                    if cur_data["original_pred"] != cur_data["after_perturb_pred"]:
                        new_data_flip_times[data_idx]+=1


            else:

                if cur_data["original_pred"]:
                    new_data_original_correctness[data_idx] += 1
                new_data_original_loss[data_idx].append(cur_data["original_loss"])
                new_data_perturbed_loss[data_idx].append(cur_data["after_perturb_loss"])

                if cur_data.__contains__("original_logit"):
                    new_data_original_logit[data_idx].append(cur_data["original_logit"])
                    new_data_perturbed_logit[data_idx].append(cur_data["after_perturb_logit"])
                    new_data_logit_diff[data_idx].append(cur_data["logit_diff"])

                    new_data_original_probability[data_idx].append(cur_data["original_probability"])
                    new_data_perturbed_probability[data_idx].append(cur_data["after_perturb_probability"])
                    new_data_probability_diff[data_idx].append(cur_data["probability_diff"])

                    new_data_golden_label[data_idx] = cur_data["golden_label"]

                if not use_normed_loss:
                    new_data_loss_diff[data_idx].append(cur_data["loss_diff"])
                else:
                    new_data_loss_diff[data_idx].append(cur_data["normed_loss_diff"])
                if use_delta_grad:
                    new_data_delta_grad[data_idx].append(cur_data["delta_grad"])
                if cur_data["original_pred"] != cur_data["after_perturb_pred"]:
                    new_data_flip_times[data_idx] += 1
            # print("tmp_idx:{}".format(tmp_idx))
            tmp_idx+=1


    # todo 转化为df之后，有一些从来没有预测对的会变成nan
    new_data_original_correctness = [new_data_original_correctness[i]/len_interval for i in range(len_dataset)]
    return new_data_loss_diff,new_data_original_correctness,new_data_flip_times,\
           new_data_delta_grad,new_data_original_loss,new_data_perturbed_loss,new_data_original_logit,\
           new_data_perturbed_logit,new_data_logit_diff ,new_data_original_probability,\
           new_data_perturbed_probability,new_data_probability_diff , new_data_golden_label

def finetune(config):
    utils.set_seed(config.seed)
    if not os.path.exists(config.output_dir):
        os.path.mkdir(parents=True)
    
    config = AutoConfig.from_pretrained(config.model_name, num_labels=config.num_labels,mirror='tuna')
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = utils.get_model(config.model_name).from_pretrained(config.model_name,config=config)
    model.to(settings.device)
    if config.reinit_classifier:
        model.reinit_classifier()
    if config.freeze_bert:
        model.freeze_Bert()

    collator = utils.Collator(pad_token_id=tokenizer.pad_token_id)
    train_dataset = utils.Huggingface_dataset_with_data_id(config, tokenizer, dataset_name =config.dataset_name, task_name=config.task_name)
    
    # 现在是两类都选择同样的ratio，一半一半，即30%+30%
    # todo  我想的是整个数据集，然后比如前30%是鲁棒的，其他是不鲁棒的
    result_data_indices, selected_label_nums = generate_data_indices(config, train_dataset,config.select_metric,config.select_ratio)
    # if config.select_metric[-2:] == "_r":
    #     select_metric2 = config.select_metric[:-2]
    # else:
    #     select_metric2 = config.select_metric + "_r"
    # result_data_indices2, selected_label_nums2 = generate_data_indices(config, train_dataset, select_metric2,
    #                                                                          1.0-config.select_ratio)
    # train_dataset.dataset = train_dataset.dataset.select(result_data_indices+result_data_indices2)


    # result_data_indices_new = result_data_indices+result_data_indices2
    # result_data_indices_new = [i for i in range(train_dataset.__len__())]
    # for i in range(len(result_data_indices_new)):
    #     data_indices_to_new_id[result_data_indices_new[i]] = i
    #     new_id_to_data_indices[i] = result_data_indices_new[i]
        # pass
    if config.dataset_name=="imdb":
        train_dataset.updata_idx(data_indices_to_new_id,new_id_to_data_indices)
    # result_data_indices = [i for i in range(len(result_data_indices))]
    # old_to_new_dict = {}


    if config.show_data > 0:
        import csv
        print("Metric:{}".format(config.select_metric))
        for i in range(config.show_data):
            print("sentence: {}, label: {}".format(train_dataset.dataset["sentence"][i],train_dataset.dataset["label"][i]))
            print()
            print()
            show_data_dir = "/root/Robust_Data/analysis_experiments/show_data/"
            show_data_file = "/root/Robust_Data/analysis_experiments/show_data/show_data.csv"
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
            cur_row.append(config.select_metric)
            cur_row.append(config.seed)
            cur_row.append(train_dataset.dataset["sentence"][i])
            cur_row.append(train_dataset.dataset["label"][i])
            cur_row.append(i)
            csv_writer.writerow(cur_row)
            out_csv.close()

        return

    print(str(selected_label_nums))
    config.selected_label_nums = str(selected_label_nums)
    train_loader = DataLoader(train_dataset, batch_size=config.bsz, shuffle=True, collate_fn=collator)
    logger.info("train dataset length: "+ str(len(train_dataset)))

    # for dev
    dev_dataset = utils.Huggingface_dataset(config, tokenizer, name_or_dataset=config.dataset_name,
                                            task_name=config.task_name, split=config.valid)
    dev_loader = DataLoader(dev_dataset, batch_size=config.eval_size, shuffle=False, collate_fn=collator)

    # for test
    if config.do_test:
        test_dataset = utils.Huggingface_dataset(config, tokenizer, name_or_dataset=config.dataset_name,
                                                 task_name=config.task_name, split='test')
        test_loader = DataLoader(test_dataset, batch_size=config.eval_size, shuffle=False, collate_fn=collator)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=config.lr,
                      eps=config.adam_epsilon,
                      correct_bias=config.bias_correction
    )


    # Use suggested learning rate scheduler
    num_training_steps = len(train_dataset) * config.epochs // config.bsz
    warmup_steps = num_training_steps * config.warmup_ratio
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_training_steps)
    if config.do_pgd_training:
        log_format = ["statistics_source",
                      "select_metric",
                      "select_ratio", "ratio_proportion",
                      "selected_label_nums",
                      "lr",
                      "seed", "epochs",
                      "pgd_step", "pgd_lr",
                      "clean", "pgd_aua", "attack_method","beta","b","pgd_adv_steps","pgd_adv_steps2","pgd_adv_lr","pgd_adv_lr2"
                      ]
    else:
        log_format = ["statistics_source",
                      "select_metric",
                      "select_ratio", "ratio_proportion",
                      "selected_label_nums",
                      "lr",
                      "seed", "epochs",
                      "pgd_step", "pgd_lr",
                      "clean", "pgd_aua", "attack_method","beta","b"
                      ]

    one_epoch_steps = int(len(train_dataset) // config.bsz)
    if config.save_steps<1 and config.save_steps>0:
        config.save_steps = int(one_epoch_steps*config.save_steps)
    save_steps = config.save_steps
    try:
        best_accuracy = 0
        global_step = 0
        for epoch in range(config.epochs):
            avg_loss = utils.ExponentialMovingAverage()

            model.train()
            # train 1 epoch
            if config.do_pgd_training > 0:
                global_step = pgd_one_epoch_new(config, avg_loss, device, epoch, model, optimizer, scheduler, train_loader,
                                                global_step, save_steps, tokenizer,
                                                config.adv_init_mag, config.adv_norm_type, config.pgd_adv_steps,
                                                config.pgd_adv_lr,
                                                config.adv_max_norm,result_data_indices=result_data_indices,adv_steps2=config.pgd_adv_steps2,adv_lr2=config.pgd_adv_lr2
                                                )
            else:
                global_step = train_one_epoch_new(config, avg_loss, device, epoch, model, optimizer, scheduler, train_loader, global_step, save_steps, tokenizer,result_data_indices,beta=config.beta,b=config.b)
            # save model
            # save_model_one_epoch(config, epoch, model, output_dir, tokenizer)
            # eval model
            if config.do_eval and not config.cal_time:
                accuracy,clean_loss = evaluate(dev_loader, device, model)
                logger.info(f'Epoch: {epoch}, '
                            f'Loss: {avg_loss.get_metric(): 0.4f}, '
                            f'Lr: {optimizer.param_groups[0]["lr"]: .3e}, '
                            f'Accuracy: {accuracy}')
                if accuracy > best_accuracy:
                    logger.info('Best performance so far.')
                    # model.save_pretrained(output_dir)
                    # tokenizer.save_pretrained(output_dir)
                    # torch.save(config, os.path.join(output_dir, "training_config.bin"))
                    best_accuracy = accuracy
                    best_dev_epoch = epoch
        if config.attack_every_step > 0:
            for step in range(num_training_steps):
                if step%config.save_steps==0:

                    one_epoch_steps = int(len(train_dataset) // config.bsz)

                    epoch = int(step//one_epoch_steps)
                    logger.info("current step:{},current epoch:{}".format(step, epoch))
                    if epoch%(config.cycle_train * 2) < config.cycle_train: #
                        logger.info("current metric:{},cureent ratio:{}".format(config.select_metric,config.select_ratio))
                        cur_select_metric = config.select_metric
                        config.cur_select_metric = cur_select_metric

                    else:
                        logger.info("current metric:{},cureent ratio:{}".format(config.select_metric2,config.select_ratio2))
                        cur_select_metric = config.select_metric2
                        config.cur_select_metric = cur_select_metric
                    config.cur_epoch = epoch
                    config.cur_step = step
                    s = Path(str(output_dir) + '/step' + str(step))
                    if config.model_name == "bert-base-uncased":
                        model = BertForSequenceClassification.from_pretrained(s, config=config)
                    elif config.model_name == "roberta-base":
                        model = RobertaForSequenceClassification.from_pretrained(s, config=config)
                    else:
                        model = BertForSequenceClassification.from_pretrained(s, config=config)
                    model.to(device)
                    if config.do_eval and not config.cal_time:
                        # todo 没改loader
                        clean,clean_loss = evaluate(dev_loader, device, model)
                        config.clean = clean

                        config.clean_loss =clean_loss.get_metric()
                    else:
                        config.clean=0
                        config.clean_loss=0
                    if config.do_pgd_attack:
                        pgd_aua,pgd_loss = do_pgd_attack( dev_loader, device, model,
                                                 adv_steps=config.pgd_step,
                                                 adv_lr=config.pgd_lr,
                                                 adv_init_mag=config.adv_init_mag,
                                                 adv_max_norm=config.adv_max_norm,
                                                 adv_norm_type=config.adv_norm_type
                                                 )
                        optimizer.zero_grad()
                        config.pgd_aua = pgd_aua
                        config.pgd_loss = pgd_loss.get_metric()
                    else:
                        config.pgd_aua = 0
                        config.pgd_loss = 0


                    do_textattack_attack(config, model, tokenizer,
                                         do_attack=config.do_attack,
                                         attack_seed=42,
                                         attack_all=config.attack_all,
                                         attack_method="textfooler",
                                         attack_every_epoch=False,
                                         attack_every_step=True,
                                         log_format=[i for i in log_format]

                                         )

        elif config.attack_every_epoch>0:
            for epoch in range(config.epochs):
                logger.info("current epoch:{}".format( epoch))
                if config.cycle_train > 0:
                    if epoch%(config.cycle_train * 2) < config.cycle_train: #
                        logger.info("current metric:{},cureent ratio:{}".format(config.select_metric,config.select_ratio))
                        cur_select_metric = config.select_metric
                        config.cur_select_metric = cur_select_metric

                    else:
                        logger.info("current metric:{},cureent ratio:{}".format(config.select_metric2,config.select_ratio2))
                        cur_select_metric = config.select_metric2
                        config.cur_select_metric = cur_select_metric
                else:
                    logger.info("current metric:{},cureent ratio:{}".format(config.select_metric, config.select_ratio))
                    cur_select_metric = config.select_metric
                    config.cur_select_metric = cur_select_metric

                config.cur_epoch = epoch
                s = Path(str(output_dir) + '/epoch' + str(epoch))
                if config.model_name == "bert-base-uncased":
                    model = BertForSequenceClassification.from_pretrained(s, config=config)
                elif config.model_name == "roberta-base":
                    model = RobertaForSequenceClassification.from_pretrained(s, config=config)
                else:
                    model = BertForSequenceClassification.from_pretrained(s, config=config)
                model.to(device)
                if config.do_eval and not config.cal_time:
                    if config.attack_every_epoch ==10:
                        config.attack_dataset_metric = config.select_metric
                        loader = DataLoader(train_dataset, batch_size=config.bsz, shuffle=True, collate_fn=collator)
                    elif config.attack_every_epoch ==20:
                        config.attack_dataset_metric = config.select_metric2
                        loader = DataLoader(train_dataset2, batch_size=config.bsz, shuffle=True, collate_fn=collator)
                    else:
                        config.attack_dataset_metric = "test_set"
                        loader = dev_loader
                    clean,clean_loss = evaluate(loader, device, model)
                    config.clean = clean
                    config.clean_loss = clean_loss.get_metric()
                else:
                    config.clean = 0
                    config.clean_loss = 0
                if config.do_pgd_attack:
                    if config.attack_every_epoch ==10:
                        config.attack_dataset_metric = config.select_metric
                        loader = DataLoader(train_dataset, batch_size=config.bsz, shuffle=True, collate_fn=collator)
                    elif config.attack_every_epoch ==20:
                        config.attack_dataset_metric = config.select_metric2
                        loader = DataLoader(train_dataset2, batch_size=config.bsz, shuffle=True, collate_fn=collator)
                    else:
                        config.attack_dataset_metric = "test_set"
                        loader = dev_loader
                    pgd_aua,pgd_loss = do_pgd_attack( loader, device, model,
                                             adv_steps=config.pgd_step,
                                             adv_lr=config.pgd_lr,
                                             adv_init_mag=config.adv_init_mag,
                                             adv_max_norm=config.adv_max_norm,
                                             adv_norm_type=config.adv_norm_type
                                             )
                    optimizer.zero_grad()
                    config.pgd_aua = pgd_aua
                    config.pgd_loss = pgd_loss.get_metric()
                else:
                    config.pgd_aua = 0
                    config.pgd_loss = 0

                do_textattack_attack(config, model, tokenizer,
                                     do_attack=config.do_attack,
                                     attack_seed=42,
                                     attack_all=config.attack_all,
                                     attack_method="textfooler",
                                     attack_every_epoch=True,
                                     attack_every_step=False,
                                     log_format=[i for i in log_format]
                                     )

        if (not config.attack_every_epoch and not config.attack_every_step and config.do_attack):
            if config.do_eval and not config.cal_time:
                config.attack_dataset_metric = "test_set"
                loader = dev_loader
                clean, clean_loss = evaluate(loader, device, model)
                config.clean = clean
                config.clean_loss = clean_loss
            else:
                config.clean = 0
                config.clean_loss = 0
            if config.do_pgd_attack:
                pgd_aua,pgd_loss = do_pgd_attack( dev_loader, device, model,
                                                 adv_steps=config.pgd_step,
                                                 adv_lr=config.pgd_lr,
                                                 adv_init_mag=config.adv_init_mag,
                                                 adv_max_norm=config.adv_max_norm,
                                                 adv_norm_type=config.adv_norm_type
                                                 )
                config.pgd_aua = pgd_aua
                config.pgd_loss = pgd_loss
            else:
                config.pgd_aua = 0
                config.pgd_loss = 0
            optimizer.zero_grad()
            do_textattack_attack(config, model, tokenizer,
                                 do_attack=config.do_attack,
                                 attack_seed=42,
                                 attack_all=config.attack_all,
                                 attack_method="textfooler",
                                 attack_every_epoch=config.attack_every_epoch,
                                 log_format=[i for i in log_format]
                                 )
        elif (not config.attack_every_epoch and not config.attack_every_step and not config.do_attack and config.do_pgd_attack):
            pgd_aua,pgd_loss = do_pgd_attack( dev_loader, device, model,
                                             adv_steps=config.pgd_step,
                                             adv_lr=config.pgd_lr,
                                             adv_init_mag=config.adv_init_mag,
                                             adv_max_norm=config.adv_max_norm,
                                             adv_norm_type=config.adv_norm_type,
                                             )
            optimizer.zero_grad()
            config.pgd_aua = pgd_aua
            config.pgd_loss = pgd_loss

    except KeyboardInterrupt:
        logger.info('Interrupted...')
