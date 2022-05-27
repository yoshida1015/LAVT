date=`date -Iseconds`

mkdir ./checkpoints/reverie/$date

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 train.py --model lavt --dataset reverie --model_id reverie --batch-size 64 --lr 0.00005 --wd 1e-2 --output-dir ./checkpoints/reverie/$date/ --swin_type tiny --epochs 40 --img_size 240 --pretrained_swin_weights pretrained_weights/swin_tiny_patch4_window7_224.pth --run_id $date 2>&1 | tee ./checkpoints/reverie/$date/train_log

