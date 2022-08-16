memo=$1

if [ -z "$memo" ]; then
   ./scripts/train.sh pretrained_weights/coat_lite_medium_384x384_f9129688.pth lite_medium 240 16 0.1 no_notification 
else
   ./scripts/train.sh pretrained_weights/coat_lite_medium_384x384_f9129688.pth lite_medium 240 16 0.1 $memo
fi
