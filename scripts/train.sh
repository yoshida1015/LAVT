date=`date -Iseconds`

mkdir -p ./checkpoints/reverie/$date

CUDA_VISIBLE_DEVICES=1,2 python3 -m torch.distributed.launch --nproc_per_node 2 --master_port 12349 train.py --model lavt --dataset reverie --model_id reverie --batch-size 16 --lr 0.00005 --wd 1e-2 --output-dir ./checkpoints/reverie/$date/ --swin_type base --epochs 40 --img_size 240 --pretrained_swin_weights pretrained_weights/swin_base_patch4_window12_384_22k.pth --run_id $date 2>&1 | tee ./checkpoints/reverie/$date/train_log

