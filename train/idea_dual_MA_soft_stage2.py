import argparse
import logging
import os
import random
import shutil
import sys
import time
from datetime import datetime
from copy import deepcopy
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 当前程序上上一级目录
sys.path.append(BASE_DIR)  # 添加环境变量

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from torch.amp import GradScaler

from dataloader.Dataset_all import *
from dataloader.transform_3D import *
from networks.net_factory_3d import net_factory_3d
from utils import losses, metrics, ramps
from val_3D import test_all_case_3D

parser = argparse.ArgumentParser()
# experiment base setting
parser.add_argument('--data_root_path', type=str,
                    default='/data/hqn_data/Data/WORD/3D/WORD-V0.1.0-Admin_cropWL_for3D',
                    help='root path')
parser.add_argument('--data_type', type=str,
                    default='Abdomen', help='Data category')
parser.add_argument('--data_name', type=str,
                    default='word_3d', help='Data name, select mode for Abdomen: word_3d')
parser.add_argument('--trainData', type=str,
                    default='trainReT04.txt', help='retrain data')
parser.add_argument('--validData', type=str,
                    default='valid.txt', help='valid data')

parser.add_argument('--model', type=str,
                    default='attention_unet_feat_2dual_3d', help=' ')
parser.add_argument('--exp', type=str,
                    default='W_weakly_SPS_soft_3d', help='experiment_name')
parser.add_argument('--fold', type=str,
                    default='stage2', help='train fold name')
parser.add_argument('--sup_type', type=str,
                    default='pseudoLab', help='supervision type, selected mode: label, scribble, pseudoLab')
parser.add_argument('--num_classes', type=int, default=17,
                    help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=20000, help='maximum epoch number to train')
parser.add_argument('--ES_interval', type=int,
                    default=20001, help='maximum iteration iternal for early-stopping')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.1,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[80, 96, 96],
                    help='patch size of network input')

parser.add_argument('--seed', type=int, default=1337, help='random seed')
args = parser.parse_args()


# 高斯函数，用于计算权重
def gaussian_weight(u, beta):
    return torch.exp(-(torch.tensor(u, dtype=torch.float32).cuda() ** beta))


def vote_threshold_label_selection_class_3D(pred1, pred2, cla1, cla2, num_classes):
    """
        input:
            pred1 & pred2: logits with per-class prediction probability: B, C, H, W
            threshold: confident predictions

        output:
            label: one label for all three branches
            mask: confident or not
    """
    pred1 = torch.softmax(pred1, dim=1)
    pred2 = torch.softmax(pred2, dim=1)
    pred1_confidence, pred1_label = pred1.max(dim=1)
    pred2_confidence, pred2_label = pred2.max(dim=1)
    same_pred = (pred1_label == pred2_label)

    # 计算每个像素的类别特定阈值
    threshold_1 = torch.zeros_like(pred1_confidence)
    threshold_2 = torch.zeros_like(pred2_confidence)

    for c in range(num_classes):
        threshold_1 = threshold_1 + cla1[c] * (pred1_label == c)
        threshold_2 = threshold_2 + cla2[c] * (pred2_label == c)

    same_pred_both_confident = same_pred * (pred1_confidence > threshold_1) * (pred2_confidence > threshold_2)
    same_pred_one_confident = same_pred * ((pred1_confidence > threshold_1) != (pred2_confidence > threshold_2))
    same_pred_no_confident = same_pred * (pred1_confidence <= threshold_1) * (pred2_confidence <= threshold_2)
    different_pred_region = (pred1_label != pred2_label)

    return same_pred_both_confident, same_pred_one_confident, same_pred_no_confident, different_pred_region


def train(args, snapshot_path):
    data_root_path = args.data_root_path
    batch_size = args.batch_size
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    trainData_txt = args.trainData
    validData_txt = args.validData
    # ES_interval = args.ES_interval

    # cla1 = [torch.from_numpy(0.5 * np.ones(batch_size)).cuda() for _ in range(num_classes)]
    # cla2 = [torch.from_numpy(0.5 * np.ones(batch_size)).cuda() for _ in range(num_classes)]
    cla1 = torch.full((num_classes,), 0.5, device='cuda')
    cla2 = torch.full((num_classes,), 0.5, device='cuda')

    model = net_factory_3d(net_type=args.model, in_chns=1, class_num=num_classes)
    model_parameter = sum(p.numel() for p in model.parameters())
    logging.info("model_parameter:{}M".format(round(model_parameter / (1024 * 1024), 2)))

    db_train = BaseDataSets(
        base_dir=data_root_path,
        split="train",
        data_txt=trainData_txt,
        transform=transforms.Compose([
            RandomCrop(args.patch_size),
            ToTensor(),
        ]),
        sup_type=args.sup_type,
        num_classes=num_classes
    )
    db_val = BaseDataSets(
        base_dir=data_root_path,
        split="val",
        data_txt=validData_txt,
        num_classes=num_classes)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss(ignore_index=num_classes)
    ce_loss2 = CrossEntropyLoss(reduction='none')

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    fresh_iter_num = iter_num
    max_epoch = max_iterations // len(trainloader) + 1
    logging.info("max epoch: {}".format(max_epoch))
    scaler = GradScaler()

    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    alpha = 0.9

    for epoch_num in iterator:
        epoch_conf_sum1 = torch.zeros(num_classes, device='cuda')
        epoch_count1 = torch.zeros(num_classes, device='cuda')
        epoch_conf_sum2 = torch.zeros(num_classes, device='cuda')
        epoch_count2 = torch.zeros(num_classes, device='cuda')
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch, gt_batch = sampled_batch['image'], sampled_batch['label'], sampled_batch['gt']
            volume_batch, label_batch, gt_batch = volume_batch.cuda(), label_batch.cuda(), gt_batch.cuda()

            with (torch.amp.autocast(device_type='cuda', dtype=torch.float16)):
                outputs, outputs_aux1 = model(volume_batch)
                outputs_soft1 = torch.softmax(outputs, dim=1)
                outputs_soft2 = torch.softmax(outputs_aux1, dim=1)

                # 获取预测结果
                pred1_confidence, pred1_label = outputs_soft1.max(dim=1)
                pred2_confidence, pred2_label = outputs_soft2.max(dim=1)

                # 累积每个类别的置信度和数量
                for c in range(num_classes):
                    mask1 = (pred1_label == c)
                    epoch_conf_sum1[c] += (pred1_confidence * mask1).sum()
                    epoch_count1[c] += mask1.sum().float()

                    mask2 = (pred2_label == c)
                    epoch_conf_sum2[c] += (pred2_confidence * mask2).sum()
                    epoch_count2[c] += mask2.sum().float()

                same_pred_both_confident, same_pred_one_confident, same_pred_no_confident, different_pred_region = vote_threshold_label_selection_class_3D(
                    outputs, outputs_aux1, cla1, cla2, num_classes)

                u = 0.3
                beta = 3

                sa_both_co_weight = gaussian_weight(3 * u, beta)
                sa_one_co_weight = gaussian_weight(2 * u, beta)
                sa_no_co_weight = gaussian_weight(u, beta)
                di_weight = gaussian_weight(0, beta)

                loss_ce1 = ce_loss(outputs, label_batch[:].long())
                loss_ce2 = ce_loss(outputs_aux1, label_batch[:].long())
                loss_ce = 0.5 * (loss_ce1 + loss_ce2)  # 式3

                temperature = 0.4
                softmax = torch.nn.Softmax(dim=1)
                output_weight = softmax(torch.cat([
                    torch.sum(outputs_soft1 * torch.log2(outputs_soft1 + 1e-12), dim=1, keepdim=True),
                    torch.sum(outputs_soft2 * torch.log2(outputs_soft2 + 1e-12), dim=1, keepdim=True),
                ], dim=1) / temperature)
                output_weight = output_weight.detach()
                pseudo_pred = output_weight[:, 0].unsqueeze(1) * outputs_soft1 + output_weight[:, 1].unsqueeze(
                    1) * outputs_soft2

                loss_pse1 = ((sa_both_co_weight * ce_loss2(outputs_soft1, pseudo_pred) * same_pred_both_confident).sum() + (
                            sa_one_co_weight * ce_loss2(outputs_soft1, pseudo_pred) * same_pred_one_confident).sum() + (
                                         sa_no_co_weight * ce_loss2(outputs_soft1,
                                                                    pseudo_pred) * same_pred_no_confident).sum() + (
                                         di_weight * ce_loss2(outputs_soft1,
                                                              pseudo_pred) * different_pred_region).sum()) / (
                                        same_pred_both_confident.sum() * sa_both_co_weight + same_pred_one_confident.sum() * sa_one_co_weight + same_pred_no_confident.sum() * sa_no_co_weight + different_pred_region.sum() * di_weight)

                loss_pse2 = ((sa_both_co_weight * ce_loss2(outputs_soft2, pseudo_pred) * same_pred_both_confident).sum() + (
                            sa_one_co_weight * ce_loss2(outputs_soft2, pseudo_pred) * same_pred_one_confident).sum() + (
                                         sa_no_co_weight * ce_loss2(outputs_soft2,
                                                                    pseudo_pred) * same_pred_no_confident).sum() + (
                                         di_weight * ce_loss2(outputs_soft2,
                                                              pseudo_pred) * different_pred_region).sum()) / (
                                        same_pred_both_confident.sum() * sa_both_co_weight + same_pred_one_confident.sum() * sa_one_co_weight + same_pred_no_confident.sum() * sa_no_co_weight + different_pred_region.sum() * di_weight)

                loss_pse_sup_soft = 0.5 * (loss_pse1 + loss_pse2)

                loss = loss_ce + 8 * loss_pse_sup_soft

            optimizer.zero_grad()
            scaler.scale(loss).backward()  # 缩放后的反向传播
            scaler.step(optimizer)
            scaler.update()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce1', loss_ce1, iter_num)
            writer.add_scalar('info/loss_ce2', loss_ce2, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_sps', loss_pse_sup_soft, iter_num)

            if iter_num > 5000 and iter_num % 200 == 0:
                cla1_str = ', '.join([f'{x:.4f}' for x in cla1])
                cla2_str = ', '.join([f'{x:.4f}' for x in cla2])
                logging.info(
                    'iteration %d : loss : %f, loss_ce: %f, loss_pse_sup_soft:%f, cla1: [%s], cla2: [%s]' %
                    (iter_num, loss.item(), loss_ce.item(), loss_pse_sup_soft.item(), cla1_str, cla2_str))

                model.eval()
                metric_list = test_all_case_3D(valloader, model, args)

                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i + 1),
                                      metric_list[class_i], iter_num)

                performance = metric_list[:, 0].mean()
                if performance > best_performance:
                    fresh_iter_num = iter_num
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                writer.add_scalar('info/val_dice_score', metric_list[:, 0].mean(), iter_num)
                logging.info("avg_metric:{} ".format(metric_list))
                logging.info('iteration %d : dice_score : %f ' % (iter_num, metric_list[:, 0].mean()))
                model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break

        with torch.no_grad():
            # 计算当前epoch的平均置信度
            current_avg_conf1 = torch.zeros(num_classes, device='cuda')
            current_avg_conf2 = torch.zeros(num_classes, device='cuda')

            for c in range(num_classes):
                if epoch_count1[c] > 0:
                    current_avg_conf1[c] = epoch_conf_sum1[c] / epoch_count1[c]
                    if (cla1[c] - current_avg_conf1[c]) > 0.3:
                        current_avg_conf1[c] = cla1[c]
                    else:
                        cla1[c] = alpha * current_avg_conf1[c] + (1 - alpha) * cla1[c]
                if epoch_count2[c] > 0:
                    current_avg_conf2[c] = epoch_conf_sum2[c] / epoch_count2[c]
                    if (cla2[c] - current_avg_conf2[c]) > 0.3:
                        current_avg_conf2[c] = cla2[c]
                    else:
                        cla2[c] = alpha * current_avg_conf2[c] + (1 - alpha) * cla2[c]


            cla1 = torch.clamp(cla1, min=0.2)
            cla2 = torch.clamp(cla2, min=0.2)

    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "/data/hqn_data/Experiment/idea_dual_metassl_stage2/{}_{}/{}_{}_{}".format(
        args.data_type, args.data_name, args.exp, args.model, args.fold)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    shutil.copyfile(
        __file__, os.path.join(snapshot_path, run_id + "_" + os.path.basename(__file__))
    )

    # logging.basicConfig(filename=snapshot_path+"/train_log.txt", level=logging.INFO,
    #                     format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logger = logging.getLogger()
    logger.handlers.clear()
    file_handler = logging.FileHandler(snapshot_path + "/train_log.txt")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)
    logging.info(str(args))
    start_time = time.time()
    train(args, snapshot_path)
    time_s = time.time() - start_time
    logging.info("time cost: {}s,i.e, {}h".format(time_s, time_s / 3600))
