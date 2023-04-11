
import torch
import torch.nn as nn
import torch.nn.functional as F
def get_delta(embedding_init,attention_mask,adv_init_mag,adv_norm_type):
    if adv_init_mag > 0:
        input_mask = attention_mask.to(embedding_init)
        input_lengths = torch.sum(input_mask, 1)
        if adv_norm_type == 'l2':
            delta = torch.zeros_like(embedding_init).uniform_(-1, 1) * input_mask.unsqueeze(
                2)
            dims = input_lengths * embedding_init.size(-1)
            magnitude = adv_init_mag / torch.sqrt(dims)
            delta = (delta * magnitude.view(-1, 1, 1))
        elif adv_norm_type == 'linf':
            delta = torch.zeros_like(embedding_init).uniform_(-adv_init_mag,
                                                                adv_init_mag) * input_mask.unsqueeze(2)
    else:
        delta = torch.zeros_like(embedding_init)
    return delta
    
def FreeLB(model, model_inputs, labels,adv_init_mag,adv_norm_type,adv_steps=7, adv_lr=1e-2, adv_max_norm=0.0, use_cur_preds=False,cur_batch_preds=None,statistics=None,data_index=0):
    word_embedding_layer = model.get_input_embeddings()
    input_ids = model_inputs['input_ids']
    attention_mask = model_inputs['attention_mask']

    embedding_init = word_embedding_layer(input_ids)
    # initialize delta
    delta = get_delta(embedding_init,attention_mask,adv_init_mag,adv_norm_type)
    total_loss = 0.0
    for astep in range(adv_steps):
        # 0. forward
        delta.requires_grad_()
        batch = {'inputs_embeds': delta + embedding_init, 'attention_mask': attention_mask}
        logits = model(**batch, return_dict=False)[0]
        _, preds = logits.max(dim=-1)
        # 1.loss backward
        if use_cur_preds:
            losses = F.cross_entropy(logits, cur_batch_preds.squeeze(-1))
        else:
            losses = F.cross_entropy(logits, labels.squeeze(-1))
        # losses = F.cross_entropy(logits, labels.squeeze(-1))
        loss = torch.mean(losses)
        loss = loss / adv_steps
        total_loss += loss.item()
        loss.backward()
        
        
        if astep == adv_steps - 1:
            return logits, preds, delta, total_loss
            for i in range(len(labels)):
                cur_logits = logits[i]
                cur_label = labels[i]
                cur_pred = preds[i]
                if use_cur_preds:
                    cur_batch_pred = cur_batch_preds[i]
                    cur_losses = F.cross_entropy(cur_logits.unsqueeze(0), cur_batch_pred.unsqueeze(0))
                else:
                    cur_losses = F.cross_entropy(cur_logits.unsqueeze(0), cur_label.unsqueeze(0))
                cur_loss = torch.mean(cur_losses)
                statistics[data_index]["after_perturb_loss"] = cur_loss.item()
                statistics[data_index]["after_perturb_pred"] = (cur_label.item() == cur_pred.item())

                statistics[data_index]["after_perturb_logit"] = cur_logits[cur_label.item()].item()
                statistics[data_index]["after_perturb_probability"] = nn.Softmax(dim=-1)(cur_logits)[
                    cur_label.item()].item()

                statistics[data_index]["logit_diff"] = statistics[data_index]["after_perturb_logit"] - statistics[data_index]["original_logit"]
                statistics[data_index]["probability_diff"] = statistics[data_index]["after_perturb_probability"] - statistics[data_index]["original_probability"]

                statistics[data_index]["loss_diff"] = statistics[data_index]["after_perturb_loss"] - statistics[data_index]["original_loss"]
                statistics[data_index]["normed_loss_diff"] = statistics[data_index]["loss_diff"] / delta.norm(p=2,dim=(1,2),keepdim=False)[i].item()
                data_index += 1
            break


        # 2. get gradient on delta
        delta_grad = delta.grad.clone().detach()

        # 3. update and clip
        if adv_norm_type == "l2":
            denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
            denorm = torch.clamp(denorm, min=1e-8)
            delta = (delta + adv_lr * delta_grad / denorm).detach()  # detach 截断反向传播?
            if adv_max_norm > 0:
                delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                exceed_mask = (delta_norm > adv_max_norm).to(embedding_init)
                reweights = (adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1,
                                                                                                    1)
                # 将权重大于config.adv_max_norm的部分更新权重为config.adv_max_norm / delta_norm * exceed_mask
                delta = (delta * reweights).detach()
        elif adv_norm_type == "linf":
            denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1,
                                                                                                        1, 1)
            denorm = torch.clamp(denorm, min=1e-8)
            delta = (delta + adv_lr * delta_grad / denorm).detach()
        embedding_init = word_embedding_layer(input_ids)  # 重新初始化embedding