date=`date -Iseconds`
model_path=$1

mkdir ./log/inf/$date

python test.py --model lavt --swin_type tiny --dataset reverie --split test --resume $model_path --workers 4 --ddp_trained_weights --img_size 240 --run_id $date 2>&1 | tee ./log/inf/$date/inf_log 

