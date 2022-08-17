import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='LAVT training and testing')
    parser.add_argument('--model_id', default='lavt', help='name to identify the model')
    parser.add_argument('--dataset', default='refcoco', help='refcoco, refcoco+, or refcocog')
    parser.add_argument('--model', default='lavt', help='model')
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('--epochs', default=40, type=int, metavar='N', 
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', 
                        help='number of data loading workers')
    parser.add_argument('--lr', default=0.00005, type=float, help='the initial learning rate')
    parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float, metavar='W', 
                        help='weight decay', dest='weight_decay')
    parser.add_argument('--amsgrad', action='store_true',
                        help='if true, set amsgrad to True in an Adam or AdamW optimizer.')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='./checkpoints/', 
                        help='path where to save checkpoint weights')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')

    # only used when testing on a single machine
    parser.add_argument('--device', default='cuda:0', help='device')  

    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--bert_tokenizer',  default='bert-base-uncased', help='BERT tokenizer')
    parser.add_argument('--ck_bert',  default='bert-base-uncased', help='pre-trained BERT weights')
    parser.add_argument('--split',  default='test', help='only used when testing')
    parser.add_argument('--refer_data_root', default='./refer/data/', 
                        help='REFER dataset root directory')
    parser.add_argument('--splitBy', default='unc', 
                        help='change to umd or google when the dataset is G-Ref (RefCOCOg)')
    parser.add_argument('--swin_type', default='base',
                        help='tiny, small, base, or large variants of the Swin Transformer')
    parser.add_argument('--coat_type', default='lite_tiny',
                        help='tiny, mini, small, lite_tiny, lite_mini, lite_samll \
                              or large variants of the CoaT')
    parser.add_argument('--pretrained_swin_weights', default='',
                        help='path to pre-trained Swin backbone weights')
    parser.add_argument('--pretrained_weights', default='',
                        help='path to pre-trained backbone weights')
    parser.add_argument('--ddp_trained_weights', action='store_true',
                        help='Only needs specified when testing,'
                             'whether the weights to be loaded are from a DDP-trained model')
    parser.add_argument('--window12', action='store_true',
                        help='only needs specified when testing,'
                             'when training, window size is inferred' 
                             'from pre-trained weights file name'
                             '(containing \'window12\').' 
                             'Initialize Swin with window size 12 instead of the default 7.')
    parser.add_argument('--mha', default='', 
                        help='If specified, should be in the format of a-b-c-d, e.g., 4-4-4-4,'
                             'where a, b, c, and d refer to the numbers of heads in stage-1,'
                             'stage-2, stage-3, and stage-4 PWAMs')
    parser.add_argument('--pin_mem', action='store_true',
                        help='If true, pin memory when using the data loader.')
    parser.add_argument('--fusion_drop', default=0.0, type=float, help='dropout rate for PWAMs')
    parser.add_argument('--img_size', default=480, type=int, help='input image size')
    parser.add_argument('--disc_data', action='store_true', help='use discount data or not')
    parser.add_argument('--visualize', action='store_true', help='visualize result or not')
    parser.add_argument('--visual_dir', default='./qualitative_images/', 
                        help='path to store visualize images')
    # Loss function
    parser.add_argument('--CE_loss_en', action='store_true', help='CE loss or not')
    parser.add_argument('--focal_enable', action='store_true', help='use Focal loss or not')
    parser.add_argument('--loss_v2', action='store_true', help='use calc_loss_v2 or not')

    # REC branch
    parser.add_argument('--rec_enable', action='store_true', help='use REC head or not')
    parser.add_argument('--box_f1_loss_en', action='store_true', help='use box F1 loss or not')
    parser.add_argument('--iou_loss_en', action='store_true', help='use IoU loss or not')
    parser.add_argument('--clip', action='store_true', help='use clip features or not')

    # wandb run id
    parser.add_argument('--memo', type=str, help='notification')
    parser.add_argument('--run_id', type=str, help='wandb run id', required=True)

    # box-supervised
    parser.add_argument('--boxinst_enable', action='store_true')
    parser.add_argument('--pairwise_size', default=3, type=int)
    parser.add_argument('--pairwise_dilation', default=2, type=int)
    parser.add_argument('--pairwise_color_thresh', default=0.3, type=float)
    parser.add_argument('--warmup', action='store_true')
    parser.add_argument('--warmup_iters', default=32, type=int)
    parser.add_argument('--mask_thr', default=0.35, type=float)
    parser.add_argument('--bkg_prj_en', action='store_true')
    parser.add_argument('--bkg_pairwise_en', action='store_true')
    parser.add_argument('--bspv_scale', default=1, type=float)

    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--size_34', action='store_true')

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args_dict = parser.parse_args()
