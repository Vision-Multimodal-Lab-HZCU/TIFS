import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))

from dataloader_ours import IcaAVELoader, exemplarLoader
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from tqdm.contrib import tzip
from model.touch_visual_model_incremental import IncreTouchVisualNet
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import random
from itertools import cycle
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def CE_loss(num_classes, logits, label):
    targets = F.one_hot(label, num_classes=num_classes)
    loss = -torch.mean(torch.sum(F.log_softmax(logits, dim=-1) * targets, dim=1))
    return loss

def l2_regularization(model, lambda_l2):
    l2_loss = 0.0
    for param in model.parameters():
        if param.requires_grad:
            l2_loss += torch.norm(param, p=2) ** 2
    return lambda_l2 * l2_loss / 2

def cal_contrastive_loss(feature_1, feature_2, temperature=0.1):
    # (BS, BS)
    score = torch.mm(feature_1, feature_2.transpose(0, 1)) / temperature
    num_sample = score.shape[0]
    label = torch.arange(num_sample).to(score.device)

    loss = CE_loss(num_sample, score, label)
    return loss

def class_contrastive_loss(feature_1, feature_2, label, temperature=0.1):
    class_matrix = label.unsqueeze(0)
    class_matrix = class_matrix.repeat(class_matrix.shape[1], 1)
    class_matrix = class_matrix == label.unsqueeze(-1)
    # (BS, BS)
    class_matrix = class_matrix.float()
    # (BS, BS)
    score = torch.mm(feature_1, feature_2.transpose(0, 1)) / temperature
    loss = -torch.mean(torch.mean(F.log_softmax(score, dim=-1) * class_matrix, dim=-1))
    return loss

def top_1_acc(logits, target):
    top1_res = logits.argmax(dim=1)
    top1_acc = torch.eq(target, top1_res).sum().float() / len(target)
    return top1_acc.item()

def adjust_learning_rate(args, optimizer, epoch):
    miles_list = np.array(args.milestones) - 1
    if epoch in miles_list:
        current_lr = optimizer.param_groups[0]['lr']
        new_lr = current_lr * 0.1
        print('Reduce lr from {} to {}'.format(current_lr, new_lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

def train(args, step, train_data_set, val_data_set, exemplar_set):
    T = 2

    train_loader = DataLoader(train_data_set, batch_size=min(args.train_batch_size, train_data_set.__len__()), num_workers=args.num_workers,
                              pin_memory=True, drop_last=True, shuffle=True)
    val_loader = DataLoader(val_data_set, batch_size=min(args.infer_batch_size, val_data_set.__len__()), num_workers=args.num_workers,
                            pin_memory=True, drop_last=False, shuffle=False)

    step_out_class_num = (step + 1) * args.class_num_per_step

    if step == 0:
        model = IncreTouchVisualNet(args, step_out_class_num)
    else:
        model = torch.load('./save/{}/step_{}_best_model.pkl'.format(args.dataset, step-1))
        model.incremental_classifier(step_out_class_num)
        old_model = torch.load('./save/{}/step_{}_best_model.pkl'.format(args.dataset, step-1))

        exemplar_loader = DataLoader(exemplar_set, batch_size=min(args.exemplar_batch_size, exemplar_set.__len__()), num_workers=args.num_workers,
                                     pin_memory=True, drop_last=True, shuffle=True)

        last_step_out_class_num = step * args.class_num_per_step

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        if step != 0:
            old_model = nn.DataParallel(old_model)

    model = model.to(device)
    if step != 0:
        old_model = old_model.to(device)
        old_model.eval()

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loss_list = []
    val_acc_list = []
    best_val_res = 0.0
    for epoch in range(args.max_epoches):
        train_loss = 0.0
        num_steps = 0
        model.train()
        if step == 0:
            iterator = tqdm(train_loader)
        else:
            iterator = tzip(train_loader, cycle(exemplar_loader))

        for samples in iterator:
            if step == 0:
                data, labels = samples
                labels = labels.to(device)
                visual = data[0]
                touch = data[1]

                visual = visual.to(device)
                touch = touch.to(device)
                out, touch_feature, visual_feature = model(visual=visual, touch=touch, out_feature_before_fusion=True)
                loss = CE_loss(step_out_class_num, out, labels)
            else:
                curr, prev = samples
                data, labels = curr
                labels = labels.to(device)
                labels_ = labels % args.class_num_per_step
                labels_ = labels_.to(device)
                exemplar_data, exemplar_labels = prev
                exemplar_labels = exemplar_labels.to(device)
                data_batch_size = labels_.shape[0]
                exemplar_data_batch_size = exemplar_labels.shape[0]

                visual = data[0]
                touch = data[1]
                exemplar_visual = exemplar_data[0]
                exemplar_touch = exemplar_data[1]
                total_visual = torch.cat((visual, exemplar_visual))
                total_touch = torch.cat((touch, exemplar_touch))
                total_visual = total_visual.to(device)
                total_touch = total_touch.to(device)
                out, touch_feature, visual_feature, spatial_attn_score, temporal_attn_score = model(visual=total_visual, touch=total_visual, out_feature_before_fusion=True, out_attn_score=True)
                with torch.no_grad():
                    old_out, old_spatial_attn_score, old_temporal_attn_score = old_model(visual=total_visual, touch=total_touch, out_attn_score=True)
                    old_out = old_out.detach()
                    old_spatial_attn_score = old_spatial_attn_score.detach()
                    old_temporal_attn_score = old_temporal_attn_score.detach()

                instance_contra_loss = cal_contrastive_loss(touch_feature, visual_feature, temperature=args.instance_contrastive_temperature)

                all_labels = torch.cat((labels, exemplar_labels))
                class_contra_loss = class_contrastive_loss(touch_feature, visual_feature, all_labels,
                                                           temperature=args.class_contrastive_temperature)

                exem_spatial_attn_score = spatial_attn_score[
                                          data_batch_size:data_batch_size + exemplar_data_batch_size].transpose(2, 3)
                exem_spatial_attn_score = exem_spatial_attn_score.reshape(-1, exem_spatial_attn_score.shape[-1])

                exem_old_spatial_attn_score = old_spatial_attn_score[
                                              data_batch_size:data_batch_size + exemplar_data_batch_size].transpose(2,
                                                                                                                    3)
                exem_old_spatial_attn_score = exem_old_spatial_attn_score.reshape(-1,
                                                                                  exem_old_spatial_attn_score.shape[-1])

                exem_temporal_attn_score = temporal_attn_score[
                                           data_batch_size:data_batch_size + exemplar_data_batch_size].transpose(1, 2)
                exem_temporal_attn_score = exem_temporal_attn_score.reshape(-1, exem_temporal_attn_score.shape[-1])

                exem_old_temporal_attn_score = old_temporal_attn_score[
                                               data_batch_size:data_batch_size + exemplar_data_batch_size].transpose(1,
                                                                                                                     2)
                exem_old_temporal_attn_score = exem_old_temporal_attn_score.reshape(-1,
                                                                                    exem_old_temporal_attn_score.shape[
                                                                                        -1])

                spatial_attn_dist_loss = F.kl_div(exem_spatial_attn_score.log(), exem_old_spatial_attn_score,
                                                  reduction='sum') / exemplar_data_batch_size
                temporal_attn_dist_loss = F.kl_div(exem_temporal_attn_score.log(), exem_old_temporal_attn_score,
                                                   reduction='sum') / exemplar_data_batch_size

                curr_out = out[:data_batch_size, last_step_out_class_num:]
                loss_curr = CE_loss(args.class_num_per_step, curr_out, labels_)

                prev_out = out[data_batch_size:data_batch_size+exemplar_data_batch_size, :last_step_out_class_num]
                loss_prev = CE_loss(last_step_out_class_num, prev_out, exemplar_labels)

                loss_CE = (loss_curr * data_batch_size + loss_prev * exemplar_data_batch_size) / (data_batch_size + exemplar_data_batch_size)

                lam_l2 = 0.05
                loss_l2 = l2_regularization(model, lam_l2)
                loss = loss_CE + lam_l2 * loss_l2
                loss += args.lam_I * instance_contra_loss
                loss += args.lam_C * class_contra_loss
                loss += args.lam * spatial_attn_dist_loss + (1 - args.lam) * temporal_attn_dist_loss
            model.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
            num_steps += 1

        train_loss /= num_steps
        train_loss_list.append(train_loss)
        print('Epoch:{} train_loss:{:.5f}'.format(epoch, train_loss), flush=True)

        all_val_out_logits = torch.Tensor([])
        all_val_labels = torch.Tensor([])
        model.eval()
        with torch.no_grad():
            for val_data, val_labels in tqdm(val_loader):
                val_visual = val_data[0]
                val_touch = val_data[1]
                val_visual = val_visual.to(device)
                val_touch = val_touch.to(device)
                if torch.cuda.device_count() > 1:
                    val_out_logits = model.module.forward(visual=val_visual, touch=val_touch)
                else:
                    val_out_logits = model(visual=val_visual, touch=val_touch)
                val_out_logits = F.softmax(val_out_logits, dim=-1).detach().cpu()
                all_val_out_logits = torch.cat((all_val_out_logits, val_out_logits), dim=0)
                all_val_labels = torch.cat((all_val_labels, val_labels), dim=0)
        val_top1 = top_1_acc(all_val_out_logits, all_val_labels)
        val_acc_list.append(val_top1)
        print('Epoch:{} val_res:{:.6f} '.format(epoch, val_top1), flush=True)

        if val_top1 > best_val_res:
            best_val_res = val_top1
            print('Saving best model at Epoch {}'.format(epoch), flush=True)
            if torch.cuda.device_count() > 1:
                torch.save(model.module, './save/{}/step_{}_best_model.pkl'.format(args.dataset, step))
            else:
                torch.save(model, './save/{}/step_{}_best_model.pkl'.format(args.dataset, step))

        plt.figure()
        plt.plot(range(len(train_loss_list)), train_loss_list, label='train_loss')
        plt.legend()
        plt.savefig('./save/fig/{}/train_loss_step_{}.png'.format(args.dataset, step))
        plt.close()

        plt.figure()
        plt.plot(range(len(val_acc_list)), val_acc_list, label='val_acc')
        plt.legend()
        plt.savefig('./save/fig/{}/val_acc_step_{}.png'.format(args.dataset, step))
        plt.close()

        if args.lr_decay and step > 0:
            adjust_learning_rate(args, opt, epoch)


def detailed_test(args, step, test_data_set, task_best_acc_list):
    print("=====================================")
    print("Start testing...")

    print("=====================================")

    model = torch.load('./save/{}/step_{}_best_model.pkl'.format(args.dataset, step))
    model.to(device)

    test_loader = DataLoader(test_data_set, batch_size=args.infer_batch_size, num_workers=args.num_workers,
                             pin_memory=True, drop_last=False, shuffle=False)

    all_test_out_logits = torch.Tensor([])
    all_test_labels = torch.Tensor([])
    model.eval()
    with torch.no_grad():
        for test_data, test_labels in tqdm(test_loader):
            test_visual = test_data[0]
            test_touch = test_data[1]
            test_visual = test_visual.to(device)
            test_touch = test_touch.to(device)
            test_out_logits = model(visual=test_visual, touch=test_touch)
            test_out_logits = F.softmax(test_out_logits, dim=-1).detach().cpu()
            all_test_out_logits = torch.cat((all_test_out_logits, test_out_logits), dim=0)
            all_test_labels = torch.cat((all_test_labels, test_labels), dim=0)
    test_top1 = top_1_acc(all_test_out_logits, all_test_labels)
    print("Incremental step {} Testing res: {:.6f}".format(step, test_top1))

    old_task_acc_list = []
    for i in range(step+1):
        step_class_list = range(i*args.class_num_per_step, (i+1)*args.class_num_per_step)
        step_class_idxs = []
        for c in step_class_list:
            idxs = np.where(all_test_labels.numpy() == c)[0].tolist()
            step_class_idxs += idxs
        step_class_idxs = np.array(step_class_idxs)
        i_labels = torch.Tensor(all_test_labels.numpy()[step_class_idxs])
        i_logits = torch.Tensor(all_test_out_logits.numpy()[step_class_idxs])
        i_acc = top_1_acc(i_logits, i_labels)
        if i == step:
            curren_step_acc = i_acc
        else:
            old_task_acc_list.append(i_acc)
    if step > 0:
        forgetting = np.mean(np.array(task_best_acc_list) - np.array(old_task_acc_list))
        print('forgetting: {:.6f}'.format(forgetting))
        for i in range(len(task_best_acc_list)):
            task_best_acc_list[i] = max(task_best_acc_list[i], old_task_acc_list[i])
    else:
        forgetting = None
    task_best_acc_list.append(curren_step_acc)

    return forgetting, test_top1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='touch_and_go')
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--infer_batch_size', type=int, default=32)
    parser.add_argument('--exemplar_batch_size', type=int, default=128)

    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_epoches', type=int, default=500)
    parser.add_argument('--num_classes', type=int, default=28)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=boolean_string, default=False)
    parser.add_argument("--milestones", type=int, default=[500], nargs='+', help="")

    parser.add_argument('--lam', type=float, default=0.5)
    parser.add_argument('--lam_I', type=float, default=0.5)
    parser.add_argument('--lam_C', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--class_num_per_step', type=int, default=7)

    parser.add_argument('--memory_size', type=int, default=340)

    parser.add_argument('--instance_contrastive_temperature', type=float, default=0.1)
    parser.add_argument('--class_contrastive_temperature', type=float, default=0.1)

    args = parser.parse_args()
    print(args)

    total_incremental_steps = args.num_classes // args.class_num_per_step

    setup_seed(args.seed)

    print('Training start time: {}'.format(datetime.now()))

    train_set = IcaAVELoader(args=args, mode='train')
    val_set = IcaAVELoader(args=args, mode='val')
    test_set = IcaAVELoader(args=args, mode='test')

    exemplar_set = exemplarLoader(args=args)

    ckpts_root = './save/{}/'.format(args.dataset)
    figs_root = './save/fig/{}/'.format(args.dataset)

    if not os.path.exists(ckpts_root):
        os.makedirs(ckpts_root)
    if not os.path.exists(figs_root):
        os.makedirs(figs_root)

    task_best_acc_list = []
    total_accuracy_list = []
    step_forgetting_list = []

    exemplar_class_vids = None
    for step in range(total_incremental_steps):
        train_set.set_incremental_step(step)
        val_set.set_incremental_step(step)
        test_set.set_incremental_step(step)

        exemplar_set._set_incremental_step_(step)

        print('Incremental step: {}'.format(step))

        train(args, step, train_set, val_set, exemplar_set)
        print("开始测试")
        step_forgetting, total_accuracy = detailed_test(args, step, test_set, task_best_acc_list)
        total_accuracy_list.append(total_accuracy)
        if step_forgetting is not None:
            step_forgetting_list.append(step_forgetting)

    Mean_forgetting = np.mean(step_forgetting_list)
    Mean_accuracy = np.mean(total_accuracy_list)
    print('Average Forgetting: {:.6f}'.format(Mean_forgetting))
    print('Average Accuracy: {:.6f}'.format(Mean_accuracy))
