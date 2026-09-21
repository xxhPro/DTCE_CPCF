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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

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
from dataloader.transform_3D_BTCV import *
from networks.net_factory_3d import net_factory_3d
from util import losses, metrics, ramps
from val_3D import test_all_case_3D

parser = argparse.ArgumentParser()
parser.add_argument('--data_root_path', type=str,
                    default='/data/hqn_data/Data/BTCV/3D/five_fold/fold4',
                    help='training data root path; subfolders: train_dir, test dir, valid_dir')
parser.add_argument('--data_type', type=str,
                    default='Abdomen', help='Data category')
parser.add_argument('--data_name', type=str,
                    default='btcv_3d', help='Data name')
parser.add_argument('--trainData', type=str,
                    default='train.txt', help='train Data, select mode: train, \
                        trainrless2, trainrlessd4, trainrlessd8, trainrlessd16')
parser.add_argument('--validData', type=str,
                    default='valid.txt', help='valid data')

parser.add_argument('--model', type=str,
                    default='attention_unet_2dual_3d', help='select mode: unet_cct_dp_3D, \
                        attention_unet_2dual_3d, unetr_2dual_3d')
parser.add_argument('--exp', type=str,
                    default='W_weakly_SPS_soft_3d', help='experiment_name')
parser.add_argument('--fold', type=str,
                    default='stage1', help='train fold name')
parser.add_argument('--sup_type', type=str,
                    default='scribble', help='supervision type, select mode: label, scribble, pseudoLab')
parser.add_argument('--num_classes', type=int, default=14,
                    help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=20000, help='maximum epoch number to train')
# parser.add_argument('--ES_interval', type=int,
#                     default=59999, help='maximum iteration iternal for early-stopping')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.1,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[80, 96, 96],
                    help='patch size of network input')
# --------
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
parser.add_argument('--thresh_warmup', type=bool,
                    default=False, help='thresh_warmup')
parser.add_argument('--conf_thresh', type=float, default=0.95, help='random seed')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
args = parser.parse_args()


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def calu_dynamic_threshold_mask(pred_u_w_mix, selected_label, num_classes, classwise_acc, conf_thresh=args.conf_thresh):
    # ## calu the mask for the consistency loss
    prob = torch.softmax(pred_u_w_mix, dim=1)
    probs_x_ulb = prob.detach()
    max_probs, max_idx = torch.max(probs_x_ulb, dim=1)

    mask = max_probs.ge(conf_thresh * classwise_acc[max_idx].squeeze(-1) /
                        (2. - classwise_acc[max_idx].squeeze(-1)))  # convex True /False
    # mask = max_probs.ge(args.conf_thresh *
    #                    (torch.log(classwise_acc[max_idx].squeeze(-1) + 1.) + 0.5)/
    #                    (math.log(2) + 0.5)).float()  # concave
    # mask = max_probs.ge(args.conf_thresh * (1 / (2. - classwise_acc[max_idx].squeeze(-1)))).float()  # low_limit
    select = max_probs.ge(args.conf_thresh)
    mask = mask.to(max_probs.dtype)  # ## 将 Ture 转换为1
    # # update
    if (select == 1).sum().item():
        selected_label[select == 1] = max_idx[select == 1]  # ## 会将selected 中的 8个图片可信的类别更新类别
    selected_label_flat = selected_label.flatten()
    counts = torch.bincount(selected_label_flat)
    pseudo_counter = defaultdict(int)
    for i, count in enumerate(counts):
        pseudo_counter[i] = count.item()

    if max(pseudo_counter.values()) < len(selected_label.flatten()):  # not all(5w) -1
        if args.thresh_warmup:
            for i in range(num_classes):
                classwise_acc[i] = pseudo_counter[i] / max(pseudo_counter.values())
        else:
            wo_negative_one = deepcopy(pseudo_counter)
            if 0 in wo_negative_one.keys():
                wo_negative_one.pop(0)
            for i in range(num_classes):
                classwise_acc[i] = pseudo_counter[i] / max(wo_negative_one.values())
    return mask


def train(args, snapshot_path):
    data_root_path = args.data_root_path
    batch_size = args.batch_size
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    trainData_txt = args.trainData
    validData_txt = args.validData
    # ES_interval = args.ES_interval

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
                             num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss(ignore_index=num_classes)
    ce_loss1 = CrossEntropyLoss()
    ce_loss2 = CrossEntropyLoss(reduction='none')
    # dice_loss = DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    fresh_iter_num = iter_num
    max_epoch = max_iterations // len(trainloader) + 1
    logging.info("max epoch: {}".format(max_epoch))

    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    scaler = GradScaler()

    selected_label = torch.zeros((args.batch_size, args.patch_size[0], args.patch_size[1], args.patch_size[2]),
                                 dtype=torch.long, ).cuda()
    classwise_acc = torch.zeros((num_classes, 1)).cuda()

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch, gt_batch = sampled_batch['image'], sampled_batch['label'], sampled_batch['gt']
            volume_batch, label_batch, gt_batch = volume_batch.cuda(), label_batch.cuda(), gt_batch.cuda()

            # B = volume_batch.size(0)  # 当前 batch 的实际样本数
            # selected_label = torch.zeros((B, args.patch_size[0], args.patch_size[1], args.patch_size[2]),
            #                              dtype=torch.long, ).cuda()

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs, outputs_aux1 = model(volume_batch)
                outputs_soft1 = torch.softmax(outputs, dim=1)
                outputs_soft2 = torch.softmax(outputs_aux1, dim=1)

                consistency_weight = get_current_consistency_weight(iter_num // 150)
                # outputs_hard1 = outputs.argmax(dim=1)
                # outputs_hard2 = outputs_aux1.argmax(dim=1)
                mask = calu_dynamic_threshold_mask(outputs, selected_label, num_classes, classwise_acc)
                # mask2 = calu_dynamic_threshold_mask(outputs_aux1, selected_label, num_classes, classwise_acc)
                # loss_dtc = dice_loss(outputs_aux1.softmax(dim=1), outputs_hard1.unsqueeze(1).float(),
                #                      ignore=(1 - mask).float())
                # loss_DTC = (1 - consistency_weight) * loss_dtc

                loss_ce1 = ce_loss(outputs, label_batch[:].long())
                loss_ce2 = ce_loss(outputs_aux1, label_batch[:].long())
                loss_ce = 0.5 * (loss_ce1 + loss_ce2)  # 式3

                alpha = random.random() + 1e-10

                soft_pseudo_label = alpha * outputs_soft1.detach() + (1.0 - alpha) * outputs_soft2.detach()
                per_pixel_ce = ce_loss2(outputs_soft1, soft_pseudo_label)
                loss_dtc = (per_pixel_ce * mask).sum() / mask.sum().clamp(min=1.0)
                loss_ce_sup = ce_loss1(outputs_soft2, soft_pseudo_label)
                loss_DTC = (1 - consistency_weight) * loss_dtc
                loss_pse_sup_soft = loss_ce_sup

                # loss = loss_DTC + loss_ce + 8.0 * loss_pse_sup_soft
                loss = 6.0 * loss_DTC + 6.0 * loss_pse_sup_soft + loss_ce

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
            writer.add_scalar('info/loss_DTC', loss_DTC, iter_num)
            writer.add_scalar('info/loss_ce1', loss_ce1, iter_num)
            writer.add_scalar('info/loss_ce2', loss_ce2, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            # writer.add_scalar('info/loss_sps', loss_pse_sup_soft, iter_num)

            if iter_num <= 14000:
                continue

            else:
                if iter_num % 200 == 0:
                    logging.info(
                        'iteration %d : loss : %f, loss_ce: %f, loss_DTC:%f, alpha: %f' %
                        (iter_num, loss.item(), loss_ce.item(), loss_DTC.item(), alpha))

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

                if iter_num % 10000 == 0:
                    save_mode_path = os.path.join(
                        snapshot_path, 'iter_' + str(iter_num) + '.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    logging.info("save model to {}".format(save_mode_path))

                # if iter_num - fresh_iter_num >= ES_interval:
                #     logging.info("early stooping since there is no model updating over 1w \
                #         iteration, iter:{} ".format(iter_num))
                #     break

                if iter_num >= max_iterations:
                    break
            # if iter_num >= max_iterations or (iter_num - fresh_iter_num >= ES_interval):
            #     iterator.close()
            #     break
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

    snapshot_path = "/data/hqn_data/Experiment/BTCV_singal_LDTCE_fold4/095/model/{}_{}/{}_{}".format(
        args.data_type, args.data_name, args.exp, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    shutil.copyfile(
        __file__, os.path.join(snapshot_path, run_id + "_" + os.path.basename(__file__))
    )

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
