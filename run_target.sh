# settings
SCRIPT_DIR=$(dirname "$(realpath "$0")")
MODEL_PATH=${SCRIPT_DIR}/model/0/model.pt
DATA_PATH=${SCRIPT_DIR}/../../tiny-imagenet/tiny-imagenet-200/
KEEP_PATH=${SCRIPT_DIR}/model/0/keep.npy
EPOCHS=100

# train and handle shadow models
for shadow_id in {0..63}; do
    python3 train.py --epochs $EPOCHS --shadow_id $shadow_id --seed $RANDOM --n_shadows 64 --savedir ${SCRIPT_DIR}/exp/tiny-imageNet --dpath $DATA_PATH --dataset tiny-imagenet --pkeep 0.5
done
python3 inference.py --savedir ${SCRIPT_DIR}/exp/tiny-imageNet --dpath $DATA_PATH --dataset tiny-imagenet
python3 score.py --savedir ${SCRIPT_DIR}/exp/tiny-imageNet --dpath $DATA_PATH --dataset tiny-imagenet


#random seed 42 83 6536 6888 65995
# for seed in 42 83 69;do
# TRAIN_TIME=$(date)
# echo -e "\n[$TRAIN_TIME] epochs $EPOCHS pkeep $PKEEP" >> "${SCRIPT_DIR}/model/0/accu.txt"
# python3 train.py --epochs $EPOCHS --savedir ${SCRIPT_DIR}/model --seed 42 --dpath $DATA_PATH --dataset tiny-imagenet --pkeep 1.0
# python3 inference.py --savedir ${SCRIPT_DIR}/model --dpath $DATA_PATH --dataset tiny-imagenet
# python3 score.py --savedir ${SCRIPT_DIR}/model --dpath $DATA_PATH --dataset tiny-imagenet
# done