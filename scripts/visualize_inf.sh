date=`date -Iseconds`

mkdir ./qualitative_images/reverie/$date

python test.py --model lavt --swin_type tiny --dataset reverie --split test --resume checkpoints/reverie/tiny_best/model_best_reverie.pth --workers 4 --ddp_trained_weights --img_size 240 --visualize --visual_dir ./qualitative_images/reverie/$date/ --run_id $date 2>&1 | tee ./qualitative_images/reverie/$date/inf_log

