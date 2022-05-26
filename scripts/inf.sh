date=`date -Iseconds`

mkdir ./log/inf/$date

python test.py --model lavt --swin_type tiny --dataset reverie --split test --resume ./checkpoints/tiny_best/model_best_reverie.pth --workers 4 --ddp_trained_weights --img_size 240 --run_id $date 2>&1 | tee ./log/inf/$date/inf_log 

