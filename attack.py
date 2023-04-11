
def do_textattack_attack(args,model, tokenizer,
                         do_attack=False,attack_seed=42,attack_all=False,
                         attack_method="textfooler",attack_every_epoch=False,attack_every_step=False,log_format=None):
    # attack_seed = 42
    model.eval()
    from attack.attack_all import do_attack_with_model

    if attack_all:
        attack_methods = ["textfooler", "textbugger",
                          "bertattack"]  # todo done bertattack放最后一个，因为要改变攻击的参数！！！
    else:
        attack_methods = [attack_method]
    # print(str(attack_all))
    # print(str(attack_methods))
    if log_format==None:
        log_format = ["statistics_source",
                       "select_metric",
                       "select_ratio", "ratio_proportion",
                       "selected_label_nums",
                       "lr",
                       "seed", "epochs",
                       "pgd_step", "pgd_lr",
                       "clean", "pgd_aua", "attack_method"
                     ]

    if attack_every_epoch:
        log_format.append("cur_epoch")
        log_format.append("cur_select_metric")
        log_format.append("cycle_train")
        log_format.append("clean_loss")
        log_format.append("pgd_loss")
        log_format.append("attack_every_epoch")
        log_format.append("attack_dataset_metric")
    elif attack_every_step:
        log_format.append("cur_epoch")
        log_format.append("save_steps")
        log_format.append("cur_step")
        log_format.append("cur_select_metric")
        log_format.append("cycle_train")
        log_format.append("clean_loss")
        log_format.append("pgd_loss")
        log_format.append("attack_every_step")
        log_format.append("attack_dataset_metric")

    for attack_method in attack_methods:
        args.attack_method = attack_method

        args_dict = args.__dict__
        data_row = [args_dict[i] for i in log_format]
        do_attack_with_model(model,tokenizer,
                             dataset_name=args.dataset_name,
                             task_name=args.task_name,valid=args.valid,
                             attack_method=args.attack_method,
                             num_examples=args.num_examples,attack_seed=42,
                             results_file=args.results_file,
                             seed=args.seed,
                             model_name=args.model_name,
                             log_format=log_format,
                             data_row=data_row
                             )
        # do_attack_with_model(args, model, tokenizer
        #                      , log_format=log_format)

    model.train()
    model.zero_grad()

def do_pgd_attack( dev_loader,
                  device, model,
                  adv_steps,adv_lr,adv_init_mag,
                  adv_norm_type,
                  adv_max_norm
                  ):
    model.eval()
    pbar = tqdm(dev_loader)
    correct = 0
    total = 0
    avg_loss = utils.ExponentialMovingAverage()
    for model_inputs, labels in pbar:
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        labels = labels.to(device)
        model.zero_grad()
        word_embedding_layer = model.get_input_embeddings()
        input_ids = model_inputs['input_ids']
        attention_mask = model_inputs['attention_mask']
        embedding_init = word_embedding_layer(input_ids)
        # initialize delta
        # if adv_init_mag > 0:
        #     input_mask = attention_mask.to(embedding_init)
        #     input_lengths = torch.sum(input_mask, 1)
        #     if adv_norm_type == 'l2':
        #         delta = torch.zeros_like(embedding_init).uniform_(-1, 1) * input_mask.unsqueeze(2)
        #         dims = input_lengths * embedding_init.size(-1)
        #         magnitude = adv_init_mag / torch.sqrt(dims)
        #         delta = (delta * magnitude.view(-1, 1, 1))
        #     elif adv_norm_type == 'linf':
        #         delta = torch.zeros_like(embedding_init).uniform_(-adv_init_mag,
        #                                                           adv_init_mag) * input_mask.unsqueeze(2)
        # else:
        delta = torch.zeros_like(embedding_init)
        total_loss = 0.0
        for astep in range(adv_steps):
            # (0) forward
            delta.requires_grad_()
            batch = {'inputs_embeds': delta + embedding_init, 'attention_mask': attention_mask}
            # logits = model(**batch).logits # todo ?不确定
            logits = model(**batch, return_dict=False)[0]  # todo ?不确定
            # _, preds = logits.max(dim=-1)
            # (1) backward
            losses = F.cross_entropy(logits, labels)
            loss = torch.mean(losses)
            # loss = loss / adv_steps
            total_loss += loss.item()
            loss.backward()
            # loss.backward(retain_graph=True)

            # (2) get gradient on delta
            delta_grad = delta.grad.clone().detach()

            # (3) update and clip
            if adv_norm_type == "l2":
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + adv_lr * delta_grad / denorm).detach()
                if adv_max_norm > 0:
                    delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                    exceed_mask = (delta_norm > adv_max_norm).to(embedding_init)
                    reweights = (adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1, 1)
                    delta = (delta * reweights).detach()
            elif adv_norm_type == "linf":
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1,
                                                                                                         1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + adv_lr * delta_grad / denorm).detach()

            model.zero_grad()
            # optimizer.zero_grad()
            embedding_init = word_embedding_layer(input_ids)
        delta.requires_grad = False
        batch = {'inputs_embeds': delta + embedding_init, 'attention_mask': attention_mask}
        model.zero_grad()
        # optimizer.zero_grad()
        logits = model(**batch).logits

        losses = F.cross_entropy(logits, labels)
        loss = torch.mean(losses)
        batch_loss = loss.item()
        avg_loss.update(batch_loss)

        _, preds = logits.max(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    pgd_accuracy = correct / (total + 1e-13)
    pgd_aua = pgd_accuracy
    logger.info(f'PGD Aua: {pgd_accuracy}')
    logger.info(f'PGD Loss: {avg_loss.get_metric()}')

    model.train()
    model.zero_grad()
    return pgd_accuracy,avg_loss
