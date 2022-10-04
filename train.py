import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn

from functools import reduce
import operator
from bert.modeling_bert import BertModel

import torchvision
from lib import segmentation

import transforms as T
#import torchvision.transforms as T
import utils
import numpy as np

import torch.nn.functional as F

import gc
from collections import OrderedDict

import wandb

#from focal_loss_pytorch.focalloss import FocalLoss
#from DiceLoss_PyTorch.loss import DiceLoss

#import clip
from skimage import color
import sys

import random

from torchinfo import summary

def torch_fix_seed(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

def dice_coefficient(x, target):
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    loss = 1. - (2 * intersection / union)
    return loss



def unfold_wo_center(x, kernel_size, dilation):
    assert x.dim() == 4
    assert kernel_size % 2 == 1

    # using SAME padding
    padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
    unfolded_x = F.unfold(
        x, kernel_size=kernel_size,
        padding=padding,
        dilation=dilation
    )

    unfolded_x = unfolded_x.reshape(
        x.size(0), x.size(1), -1, x.size(2), x.size(3)
    )

    # remove the center pixels
    size = kernel_size ** 2
    unfolded_x = torch.cat((
        unfolded_x[:, :, :size // 2],
        unfolded_x[:, :, size // 2 + 1:]
    ), dim=2)

    return unfolded_x


def compute_project_term(mask_scores, gt_bitmasks):
    mask_losses_y = dice_coefficient(
        mask_scores.max(dim=2, keepdim=True)[0],
        gt_bitmasks.max(dim=2, keepdim=True)[0]
    )
    mask_losses_x = dice_coefficient(
        mask_scores.max(dim=3, keepdim=True)[0],
        gt_bitmasks.max(dim=3, keepdim=True)[0]
    )
    return (mask_losses_x + mask_losses_y).mean()


def compute_pairwise_term(mask_logits, pairwise_size, pairwise_dilation):
    assert mask_logits.dim() == 4

    log_fg_prob = F.logsigmoid(mask_logits)
    log_bg_prob = F.logsigmoid(-mask_logits)

    log_fg_prob_unfold = unfold_wo_center(
        log_fg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )
    log_bg_prob_unfold = unfold_wo_center(
        log_bg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )

    # the probability of making the same prediction = p_i * p_j + (1 - p_i) * (1 - p_j)
    # we compute the the probability in log space to avoid numerical instability
    log_same_fg_prob = log_fg_prob[:, :, None] + log_fg_prob_unfold
    log_same_bg_prob = log_bg_prob[:, :, None] + log_bg_prob_unfold

    max_ = torch.max(log_same_fg_prob, log_same_bg_prob)
    log_same_prob = torch.log(
        torch.exp(log_same_fg_prob - max_) +
        torch.exp(log_same_bg_prob - max_)
    ) + max_

    # loss = -log(prob)
    return -log_same_prob[:, 0]

def get_images_color_similarity(images, image_masks, kernel_size, dilation):
    assert images.dim() == 4
    assert images.size(0) == 1

    unfolded_images = unfold_wo_center(
        images, kernel_size=kernel_size, dilation=dilation
    )

    diff = images[:, :, None] - unfolded_images
    similarity = torch.exp(-torch.norm(diff, dim=1) * 0.5).cuda()

    unfolded_weights = unfold_wo_center(
        image_masks[None, None], kernel_size=kernel_size,
        dilation=dilation
    )
    unfolded_weights = torch.max(unfolded_weights, dim=1)[0].cuda()

    return similarity * unfolded_weights

def add_bitmasks_from_boxes(args, images, image_masks, im_h, im_w, bboxes):
    #stride = args.mask_out_stride
    #print(f"stride: {stride}")
    #start = int(stride // 2)

    #assert images.size(2) % stride == 0
    #assert images.size(3) % stride == 0

    #downsampled_images = F.avg_pool2d(
    #    images.float(), kernel_size=stride,
    #    stride=stride, padding=0
    #)[:, [2, 1, 0]]
    #print(f"downsampled_images: {downsampled_images.size()}")
    #image_masks = image_masks[:, start::stride, start::stride]
    #print(f"image_masks: {image_masks.size()}")

    for im_i in range(args.batch_size):
        images_lab = color.rgb2lab(images[im_i].byte().permute(1, 2, 0).cpu().numpy())
        images_lab = torch.as_tensor(images_lab, device=images.device, dtype=torch.float32)
        images_lab = images_lab.permute(2, 0, 1)[None]
        images_color_similarity = get_images_color_similarity(
            images_lab, image_masks[im_i],
            args.pairwise_size, args.pairwise_dilation
        )
        per_box = bboxes[im_i]
        # only one bbox pre image in RES
        bitmask_full = torch.zeros((im_h, im_w)).float().cuda()
        bitmask_full[int(im_h * per_box[1]):int(im_h * per_box[3] + 1), int(im_w * per_box[0]):int(im_w * per_box[2] + 1)] = 1.0
        # already downsampled in dataloader
        bitmask = bitmask_full
        per_im_bitmasks = torch.unsqueeze(bitmask, 0)
        per_im_bitmasks_full = torch.unsqueeze(bitmask_full, 0)

        if im_i == 0:
            gt_bitmasks = per_im_bitmasks
            gt_bitmasks_full = per_im_bitmasks_full
            color_similarity = images_color_similarity
        else:
            gt_bitmasks = torch.cat([gt_bitmasks, per_im_bitmasks], dim=0)
            gt_bitmasks_full = torch.cat([gt_bitmasks_full, per_im_bitmasks_full], dim=0)
            color_similarity = torch.cat([color_similarity, images_color_similarity], dim=0)
    per_im_gt = dict()
    per_im_gt["bitmasks"] = torch.unsqueeze(gt_bitmasks, 1)
    per_im_gt["bitmasks_full"] = torch.unsqueeze(gt_bitmasks_full, 1)
    per_im_gt["color_similarity"] = color_similarity
    return per_im_gt


        #per_im_gt_inst.gt_bitmasks = torch.stack(per_im_bitmasks, dim=0)
        #per_im_gt_inst.gt_bitmasks_full = torch.stack(per_im_bitmasks_full, dim=0)
        #per_im_gt_inst.image_color_similarity = torch.cat([
        #    images_color_similarity for _ in range(len(per_im_gt_inst))
        #], dim=0)
        #print(f"per_im_gt_inst.gt_bitmasks: {per_im_gt_inst.gt_bitmasks.size()}")
        #print(f"per_im_gt_inst.gt_bitmasks_full: {per_im_gt_inst.gt_bitmasks_full.size()}")
        #print(f"per_im_gt_inst.image_color_similarity: {per_im_gt_inst.image_color_similarity.size()}")

def get_dataset(image_set, transform, args):
    from data.dataset_refer_bert import ReferDataset
    ds = ReferDataset(args,
                      split=image_set,
                      image_transforms=transform,
                      target_transforms=None
                      )
    num_classes = 2

    return ds, num_classes

# IoU calculation for validation
def binary_IoU(pred, gt, thr):
    #pred = pred.argmax(1)
    pred = (pred > thr).float()

    intersection = torch.sum(torch.mul(pred, gt))
    union = torch.sum(torch.add(pred, gt)) - intersection

    if intersection == 0 or union == 0:
        iou = 0
    else:
        iou = float(intersection) / float(union)

    return iou, intersection, union

# IoU calculation for validation
def IoU(pred, gt):
    pred = pred.argmax(1)

    intersection = torch.sum(torch.mul(pred, gt))
    union = torch.sum(torch.add(pred, gt)) - intersection

    if intersection == 0 or union == 0:
        iou = 0
    else:
        iou = float(intersection) / float(union)

    return iou, intersection, union

def bb_iou(pred_bb, true_bb):
    x1_pred, y1_pred, x2_pred, y2_pred = pred_bb
    x1_true, y1_true, x2_true, y2_true = true_bb

    area_true = (x2_true - x1_true) * (y2_true - y1_true)
    area_pred = (x2_pred - x1_pred) * (y2_pred - y1_pred)
    x1_iou = max(x1_pred, x1_true)
    y1_iou = max(y1_pred, y1_true)
    x2_iou = min(x2_pred, x2_true)
    y2_iou = min(y2_pred, y2_true)
    w_iou = x2_iou - x1_iou
    h_iou = y2_iou - y1_iou

    if w_iou < 0 or h_iou < 0:
        return 0.0
    else:
        area_iou = w_iou * h_iou
        iou = area_iou / (area_pred + area_true - area_iou)
        return iou

def IoU_loss(pred, gt):
    B, bbox = pred.size()
    loss = 0.
    #for i in range(B):
        #loss += bb_iou(pred[i], gt[i])
    loss = torchvision.ops.box_iou(pred, gt)
    #1. - (loss / B)
    return 1 - loss.mean()

def get_transform(args):
    #transforms = [T.Resize((args.img_size, args.img_size), interpolation=T.InterpolationMode.BICUBIC),
    if args.size_34:
        transforms = [T.Resize(args.img_size, args.img_size*640/480),
                      T.ToTensor(),
                      T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                      ]
    else:
        transforms = [T.Resize(args.img_size, args.img_size),
                      T.ToTensor(),
                      T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                      ]

    return T.Compose(transforms)


def criterion(input, target):
    weight = torch.FloatTensor([0.9, 1.1]).cuda()
    return nn.functional.cross_entropy(input, target, weight=weight)

def calc_loss(input, target, loss_func):
    bkg, obj = torch.split(input, 1, dim=1)
    loss = loss_func(bkg.squeeze(), (target-1)*-1) + loss_func(obj.squeeze(), target)
    return loss

def calc_loss_v2(input, target, loss_func):
    bkg, obj = torch.split(input, 1, dim=1)
    loss = loss_func((1-bkg).squeeze(), target) + loss_func(obj.squeeze(), target)
    return loss

def box_f1_loss(pred, gt):
    loss = torch.sum(torch.square(pred - gt))
    loss /= 100
    return loss

##### dice loss ####
#def dice_loss(pred, target, smooth = 1.):
#    pred = pred.contiguous()
#    target = target.contiguous()
#    intersection = (pred * target).sum(dim=2).sum(dim=2)
#    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
#    return loss.mean()
#
##### dice + CE ####
#def dice_ce_loss(pred, target, metrics=None, bce_weight=0.5):
#    # Dice LossとCategorical Cross Entropyを混ぜていい感じにしている
#    bce = F.binary_cross_entropy_with_logits(pred, target)
#    pred = torch.sigmoid(pred)
#    dice = dice_loss(pred, target)
#    loss = bce * bce_weight + dice * (1 - bce_weight)
#    return loss

def dice_loss(inputs, targets):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    #inputs = inputs.flatten(0)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum()

def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs.float(), targets.float(), reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum()

def evaluate(model, data_loader, bert_model, use_clip, mask_thr):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    total_its = 0
    acc_ious = 0

    # evaluation variables
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            total_its += 1
            image, target, sentences, attentions, _, _ = data
            image, target, sentences, attentions = image.cuda(non_blocking=True),\
                                                   target.cuda(non_blocking=True),\
                                                   sentences.cuda(non_blocking=True),\
                                                   attentions.cuda(non_blocking=True)

            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)

            if use_clip:
                last_hidden_states = bert_model(sentences)  # (6, 10, 512)
            else:
                last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]  # (6, 10, 768)
            embedding = last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
            attentions = attentions.unsqueeze(dim=-1)  # (B, N_l, 1)
            output, bbox = model(image, embedding, l_mask=attentions)
            iou, I, U = IoU(output, target)
            #iou, I, U = binary_IoU(output, target, mask_thr)
            acc_ious += iou
            mean_IoU.append(iou)
            cum_I += I
            cum_U += U
            for n_eval_iou in range(len(eval_seg_iou_list)):
                eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                seg_correct[n_eval_iou] += (iou >= eval_seg_iou)
            seg_total += 1
        iou = acc_ious / total_its

    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU)
    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU * 100.))
    results_str = ''
    prec_res = dict()
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
        prec_res[str(eval_seg_iou_list[n_eval_iou])] = seg_correct[n_eval_iou] * 100. / seg_total
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
    print(results_str)

    return 100 * iou, 100 * cum_I / cum_U, mIoU * 100., prec_res


def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, epoch, print_freq,
                    iterations, bert_model, args):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    train_loss = 0
    total_its = 0

    for data in metric_logger.log_every(data_loader, print_freq, header):
        total_its += 1
        image, target, sentences, attentions, _, bbox_gt = data
        image, target, sentences, attentions, bbox_gt = image.cuda(non_blocking=True),\
                                                        target.cuda(non_blocking=True),\
                                                        sentences.cuda(non_blocking=True),\
                                                        attentions.cuda(non_blocking=True),\
                                                        bbox_gt.cuda(non_blocking=True)

        sentences = sentences.squeeze(1)
        attentions = attentions.squeeze(1)

        if args.clip:
            last_hidden_states = bert_model(sentences)  # (6, 10, 512)
        else:
            last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]  # (6, 10, 768)
        embedding = last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
        attentions = attentions.unsqueeze(dim=-1)  # (batch, N_l, 1)

        if args.rec_enable == True:
            output, bbox = model(image, embedding, l_mask=attentions, rec_enable=True)
            #summary(model, input_data=[image, embedding, attentions])
            bb_loss = 0
            if args.box_f1_loss_en:
                bb_loss += box_f1_loss(bbox, bbox_gt)
            if args.iou_loss_en:
                bb_loss += 0.05 * IoU_loss(bbox, bbox_gt)
            #print(f"bb_loss rate:{rec_loss(bbox, bbox_gt)/(0.05*IoU_loss(bbox, bbox_gt))}")
        else:
            output, _ = model(image, embedding, l_mask=attentions, rec_enable=False)
            bb_loss = 0

        if args.boxinst_enable == True:
            #original_image_mask = [torch.ones_like(x[0], dtype=torch.float32) for x in image]
            ####for i in range(len(original_image_masks)):
            ####    im_h = batched_inputs[i]["height"]
            ####    pixels_removed = int(
            ####        self.bottom_pixels_removed *
            ####        float(original_images[i].size(1)) / float(im_h)
            ####    )
            ####    if pixels_removed > 0:
            ####        original_image_masks[i][-pixels_removed:, :] = 0
            #original_mask_tensor = torch.tensor(original_image_mask)

            original_mask_tensor = torch.ones([image.size(0), image.size(2), image.size(3)], dtype=torch.float32)

            per_im_gt = add_bitmasks_from_boxes(
                args, image, original_mask_tensor,
                image.size(-2), image.size(-1), bbox_gt
            )

            bkg, obj = torch.split(output, 1, dim=1)
            bkg_scores = (1.0 - bkg).sigmoid()
            obj_scores = obj.sigmoid()

            if args.bkg_prj_en:
                b_loss_prj_term = compute_project_term(bkg_scores, per_im_gt["bitmasks"])
            else:
                b_loss_prj_term = 0

            if args.bkg_pairwise_en:
                b_pairwise_losses = compute_pairwise_term(
                    bkg, args.pairwise_size,
                    args.pairwise_dilation
                )
                weights = (per_im_gt["color_similarity"] >= args.pairwise_color_thresh).float() * per_im_gt["bitmasks"].float()
                b_loss_pairwise = (b_pairwise_losses * weights).sum() / weights.sum().clamp(min=1.0)
                if args.warmup:
                    warmup_factor = min(float(total_its) / float(args.warmup_iters), 1.0)
                else:
                    warmup_factor =  1.0
                b_loss_pairwise = b_loss_pairwise * warmup_factor
            else:
                b_loss_pairwise = 0


            o_loss_prj_term = compute_project_term(obj_scores, per_im_gt["bitmasks"])

            o_pairwise_losses = compute_pairwise_term(
                obj, args.pairwise_size,
                args.pairwise_dilation
            )

            weights = (per_im_gt["color_similarity"] >= args.pairwise_color_thresh).float() * per_im_gt["bitmasks"].float()
            o_loss_pairwise = (o_pairwise_losses * weights).sum() / weights.sum().clamp(min=1.0)
            if args.warmup:
                warmup_factor = min(float(total_its) / float(args.warmup_iters), 1.0)
            else:
                warmup_factor =  1.0
            o_loss_pairwise = o_loss_pairwise * warmup_factor

            box_sp_loss = b_loss_prj_term + b_loss_pairwise + o_loss_prj_term + o_loss_pairwise 
        else:
            box_sp_loss = 0
            
        #torch.autograd.set_detect_anomaly(True)
        if args.CE_loss_en:
            loss = criterion(output, target) 
        else:
            loss = 0
        if args.focal_enable:
            if not args.loss_v2:
                loss += 0.001 * calc_loss(output, target, sigmoid_focal_loss)
            else:
                loss += 0.001 * calc_loss_v2(output, target, sigmoid_focal_loss)
        loss += bb_loss*20
        loss += box_sp_loss*args.bspv_scale
        optimizer.zero_grad()  # set_to_none=True is only available in pytorch 1.6+
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        wandb.log({'train_loss':loss.item()})

        train_loss += loss.item()
        iterations += 1
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

        del image, target, sentences, attentions, loss, output, data, last_hidden_states, embedding
        gc.collect()
        torch.cuda.empty_cache()

def main(args):
    torch_fix_seed(args.seed)
    print(args.memo)
    runID = args.memo + "_" + args.run_id
    wandb.init(name=runID, project='lavt')

    dataset, num_classes = get_dataset("train",
                                       get_transform(args=args),
                                       args=args)
    dataset_test, _ = get_dataset("test",
                                  get_transform(args=args),
                                  args=args)

    print(f"local rank {args.local_rank} / global rank {utils.get_rank()} successfully built train dataset.")
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank,
                                                                    shuffle=True)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=args.pin_mem, drop_last=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers)

    print(args.model)
    model = segmentation.__dict__[args.model](pretrained=args.pretrained_weights,
                                              args=args)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
    single_model = model.module

    model_class = BertModel
    bert_model = model_class.from_pretrained(args.ck_bert)
    bert_model.pooler = None  # a work-around for a bug in Transformers = 3.0.2 that appears for DistributedDataParallel
    bert_model.cuda()
    bert_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(bert_model)
    bert_model = torch.nn.parallel.DistributedDataParallel(bert_model, device_ids=[args.local_rank])
    single_bert_model = bert_model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        single_model.load_state_dict(checkpoint['model'])
        single_bert_model.load_state_dict(checkpoint['bert_model'])
    backbone_no_decay = list()
    backbone_decay = list()
    for name, m in single_model.backbone.named_parameters():
        if 'norm' in name or 'absolute_pos_embed' in name or 'relative_position_bias_table' in name:
            backbone_no_decay.append(m)
        else:
            backbone_decay.append(m)

    params_to_optimize = [
        {'params': backbone_no_decay, 'weight_decay': 0.0},
        {'params': backbone_decay},
        {"params": [p for p in single_model.classifier.parameters() if p.requires_grad]},
        # the following are the parameters of bert
        {"params": reduce(operator.concat,
                          [[p for p in single_bert_model.encoder.layer[i].parameters()
                            if p.requires_grad] for i in range(10)])},
    ]

    optimizer = torch.optim.AdamW(params_to_optimize,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  amsgrad=args.amsgrad
                                  )

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                     lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9)

    start_time = time.time()
    iterations = 0
    best_oIoU = -0.1

    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        resume_epoch = checkpoint['epoch']
    else:
        resume_epoch = -999

    if args.clip:
        clip_model, _ = clip.load("RN50", device=device)

    for epoch in range(max(0, resume_epoch+1), args.epochs):
        data_loader.sampler.set_epoch(epoch)
        if args.clip:
            train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, epoch, args.print_freq,
                        iterations, clip_model, args)
            iou, overallIoU, mIoU, prec = evaluate(model, data_loader_test, clip_model, args.clip, args.mask_thr)
        else:
            train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, epoch, args.print_freq,
                        iterations, bert_model, args)
            iou, overallIoU, mIoU, prec = evaluate(model, data_loader_test, bert_model, args.clip, args.mask_thr)

        #iou, overallIoU = evaluate(model, data_loader_test, bert_model)
        wandb.log({'oIoU':overallIoU})
        wandb.log({'mIoU':mIoU})
        wandb.log({'P@0.5':prec["0.5"]})
        wandb.log({'P@0.6':prec["0.6"]})
        wandb.log({'P@0.7':prec["0.7"]})
        wandb.log({'P@0.8':prec["0.8"]})
        wandb.log({'P@0.9':prec["0.9"]})

        print('Average object IoU {}'.format(iou))
        print('Overall IoU {}'.format(overallIoU))
        #save_checkpoint = (best_oIoU < overallIoU)
        #if save_checkpoint:
        if True:
            print('Better epoch: {}\n'.format(epoch))
            dict_to_save = {'model': single_model.state_dict(), 'bert_model': single_bert_model.state_dict(),
                            'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': args,
                            'lr_scheduler': lr_scheduler.state_dict()}
            utils.save_on_master(dict_to_save, os.path.join(args.output_dir,
                                                            'model_best_{}.pth'.format(args.model_id)))
            best_oIoU = overallIoU

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    # set up distributed learning
    utils.init_distributed_mode(args)
    print('Image size: {}'.format(str(args.img_size)))
    main(args)
