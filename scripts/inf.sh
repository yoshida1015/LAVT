date=`date -Iseconds`
checkpoint=$1

mkdir -p ./log/inf/$date

python test.py --model lavt --swin_type base --dataset reverie --model_id reverie --split test --resume $checkpoint --workers 4 --ddp_trained_weights --window12 --img_size 240 --run_id $date 2>&1 | tee ./log/inf/$date/inf_log  

