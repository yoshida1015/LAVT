import datetime
import os
import os.path as osp
import time

import torch
import torch.utils.data
from torch import nn

from bert.modeling_bert import BertModel
import torchvision

from lib import segmentation
import transforms as T
import utils

import numpy as np
from PIL import Image, ImageDraw
import torch.nn.functional as F


from bert.tokenization_bert import BertTokenizer
from args import get_parser
import matplotlib.pyplot as plt


def draw_bb(img_path, bbox, output_pass):
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    #x, y, w, h = bbox
    #x1, y1 = x+w, y+h
    x, y, x1, y1 = bbox
    draw.rectangle((x, y, x1, y1), outline=(255, 0, 0), width=2)
    img.save(output_pass)
    

def get_dataset(image_set, transform, args):
    from data.dataset_refer_bert import ReferDataset
    ds = ReferDataset(args,
                      split=image_set,
                      image_transforms=transform,
                      target_transforms=None,
                      eval_mode=True
                      )
    num_classes = 2
    return ds, num_classes


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def unnorm(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return img 

def evaluate(model, data_loader, bert_model, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)               
    count = 0
    s_count = 0
    # evaluation variables
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []
    header = 'Test:'

    #cnt=0
    #IMAGE_DIR = osp.join(args.refer_data_root, 'images/reverie_images/')
    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            image, target, sentences, attentions, img_fname, bbox = data
            image, target, sentences, attentions = image.to(device), target.to(device), \
                                                   sentences.to(device), attentions.to(device)
            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)
            target = target.cpu().data.numpy()
            for j in range(sentences.size(-1)):
                last_hidden_states = bert_model(sentences[:, :, j], attention_mask=attentions[:, :, j])[0]
                embedding = last_hidden_states.permute(0, 2, 1)
                output, _ = model(image, embedding, l_mask=attentions[:, :, j].unsqueeze(-1))
                output = output.cpu()
                output_mask = output.argmax(1).data.numpy()

                I, U = computeIoU(output_mask, target)
                if U == 0:
                    this_iou = 0.0
                else:
                    this_iou = I*1.0/U

                ### save img and inst
                if args.visualize:
                    inst = tokenizer.decode(sentences[:, :, j][0])
                    image_name = str(count)+ "_" + img_fname[0][5:-4]
                    imshow(torchvision.utils.make_grid(unnorm(image[0].cpu())))
                    plt.axis("off")
                    plt.savefig(args.visual_dir + image_name, bbox_inches="tight", pad_inches=0.0)
                    plt.imshow(output_mask[0])
                    plt.axis("off")
                    plt.savefig(args.visual_dir + image_name[:-4] + "_mask" + ".jpg", \
                        bbox_inches="tight", pad_inches=0.0)
                    plt.imshow(target[0])
                    plt.axis("off")
                    plt.savefig(args.visual_dir + image_name[:-4] + "_targ" + ".jpg", \
                        bbox_inches="tight", pad_inches=0.0)
                    count += 1

                    with open(f"{args.visual_dir}res_inst.txt", mode='a') as f:
                        print(f"{image_name};iou:{this_iou};sentence:{inst}", file=f)

                #output_fname = str(cnt) + "_" + img_fname[0][5:]
                #draw_bb(IMAGE_DIR + img_fname[0], bbox, "bbox_img/" + output_fname)
                #cnt+=1
                #with open(f"bbox_img/inst.txt", mode='a') as f:
                #    inst = tokenizer.decode(sentences[:, :, j][0])
                #    print(f"{output_fname}:{inst}", file=f)

                
                mean_IoU.append(this_iou)
                cum_I += I
                cum_U += U
                for n_eval_iou in range(len(eval_seg_iou_list)):
                    eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                    seg_correct[n_eval_iou] += (this_iou >= eval_seg_iou)
                seg_total += 1

            del image, target, sentences, attentions, last_hidden_states, embedding, output, output_mask

    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU)
    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU*100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
    print(results_str)


def get_transform(args):
    transforms = [#T.Resize(args.img_size, args.img_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ]

    return T.Compose(transforms)


def computeIoU(pred_seg, gd_seg):
    I = np.sum(np.logical_and(pred_seg, gd_seg))
    U = np.sum(np.logical_or(pred_seg, gd_seg))

    return I, U


def main(args):
    device = torch.device(args.device)
    dataset_test, _ = get_dataset(args.split, get_transform(args=args), args)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                                   sampler=test_sampler, num_workers=args.workers)
    print(args.model)
    single_model = segmentation.__dict__[args.model](pretrained='',
                                                     args=args)
    single_model.to(device)
    model_class = BertModel
    single_bert_model = model_class.from_pretrained(args.ck_bert)
    # work-around for a transformers bug; need to update to a newer version of transformers to remove these two lines
    if args.ddp_trained_weights:
        single_bert_model.pooler = None
    checkpoint = torch.load(args.resume, map_location='cpu')
    single_bert_model.load_state_dict(checkpoint['bert_model'])
    single_model.load_state_dict(checkpoint['model'])
    model = single_model.to(device)
    bert_model = single_bert_model.to(device)

    evaluate(model, data_loader_test, bert_model, device=device)


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    print('Image size: {}'.format(str(args.img_size)))
    main(args)
