date=`date -Iseconds`
model_path=$1
coat_type=$2
img_size=$3

mkdir -p ./log/inf/$date

python test.py --model lavt --coat_type $coat_type --dataset reverie --split test --resume $model_path --workers 4 --ddp_trained_weights --img_size $img_size --run_id $date 2>&1 | tee ./log/inf/$date/inf_log 

