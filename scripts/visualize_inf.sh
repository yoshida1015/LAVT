date=`date -Iseconds`
model_path=$1
coat_type=lite_medium
img_size=480

mkdir ./qualitative_images/reverie/$date

python test.py --model lavt --coat_type $coat_type --dataset reverie --split test --resume $model_path --workers 4 --ddp_trained_weights --img_size $img_size --visualize --visual_dir ./qualitative_images/reverie/$date/ --run_id $date 2>&1 | tee ./qualitative_images/reverie/$date/inf_log

