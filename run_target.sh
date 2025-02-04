# settings
SCRIPT_DIR=$(dirname "$(realpath "$0")")
MODEL_PATH=${SCRIPT_DIR}/model/0/model.pt
DATA_PATH=${SCRIPT_DIR}/../cifar100/dataset/
KEEP_PATH=${SCRIPT_DIR}/model/0/keep.npy
EPOCHS=65
#random seed 42 83 6536 6888 65995
for seed in 42 83 6536 6888 65995;do
    python3 train.py --epochs $EPOCHS --savedir ${SCRIPT_DIR}/model --seed $seed
    python3 inference.py --savedir ${SCRIPT_DIR}/model
    python3 score.py --savedir ${SCRIPT_DIR}/model
    python3 plot.py --keep $KEEP_PATH --scores ${SCRIPT_DIR}/model/0/scores.npy --name FULL
done