date=`date -Iseconds`
pretrain=$1
coat_type=$2
img_size=$3
batch_size=$4
thr=$5

mkdir -p ./checkpoints/reverie/$date

if [ -z "$pretrain" ]; then
    CUDA_VISIBLE_DEVICES=0,1,2 python3 -m torch.distributed.launch --nproc_per_node 3 --master_port 12350 train.py --model lavt --dataset reverie --model_id reverie --batch-size $batch_size --lr 0.00005 --wd 1e-2 --output-dir ./checkpoints/reverie/$date/ --coat_type $coat_type --epochs 40 --img_size $img_size --use_bbox --boxinst --pairwise_color_thresh $thr --run_id $date 2>&1 | tee ./checkpoints/reverie/$date/train_log
else
    CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node 1 --master_port 12366 train.py --model lavt --dataset reverie --model_id reverie --batch-size $batch_size --lr 0.00005 --wd 1e-2 --output-dir ./checkpoints/reverie/$date/ --coat_type $coat_type --pretrained_weights $pretrain --epochs 40 --img_size $img_size --pairwise_color_thresh $thr --run_id $date 2>&1 | tee ./checkpoints/reverie/$date/train_log
fi




