date=`date -Iseconds`
pretrain=$1

mkdir ./checkpoints/reverie/$date

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 train.py --model lavt --dataset reverie --model_id reverie --batch-size 64 --lr 0.00005 --wd 1e-2 --output-dir ./checkpoints/reverie/$date/ --coat_type lite_tiny --pretrained_weights $pretrain --epochs 40 --img_size 240 --run_id $date 2>&1 | tee ./checkpoints/reverie/$date/train_log

