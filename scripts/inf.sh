date=`date -Iseconds`

mkdir ./log/inf/$date

python test.py --model lavt --swin_type base --dataset reverie --model_id reverie --split test --resume ./from_abci/2022-07-23T03:16:07+0900/model_best_reverie.pth --workers 4 --img_size 240 --run_id $date 2>&1 | tee ./log/inf/$date/inf_log 

#--ddp_trained_weights 
