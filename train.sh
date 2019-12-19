mkdir -p logs
export CUDA_VISIBLE_DEVICES=$1
echo Using GPU Device "$1"
export PYTHONUNBUFFERED="True"

config_key='rec_resnet_sunrgbd'  # infomax_resnet_sunrgbd, seg_resnet_sunrgbd, seg_resnet_cityscapes, rec_resnet_sunrgbd, rec_resnet_nyud2, rec_resnet_mit67

LOG="logs/traininginfo.$config_key.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging to "$LOG"

starttime=`date +'%Y-%m-%d %H:%M:%S'`
cat train.sh

for i in $(seq 1 1)
do
    python train.py $1 $config_key

done

endtime=`date +'%Y-%m-%d %H:%M:%S'`
start_seconds=$(date --date="$starttime" +%s);
end_seconds=$(date --date="$endtime" +%s);
echo "start time:"$((end_seconds-start_seconds))"s"
echo "------------"
