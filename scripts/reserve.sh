date=`date -Iseconds`

mkdir ./checkpoints/reverie/$date

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 train.py --model lavt --dataset reverie --model_id reverie --batch-size 8 --lr 0.00005 --wd 1e-2 --output-dir ./checkpoints/reverie/$date/ --swin_type large --epochs 40 --img_size 240 --pretrained_swin_weights pretrained_weights/swin_large_patch4_window12_384_22kto1k.pth --run_id $date 2>&1 | tee ./checkpoints/reverie/$date/train_log

