# settings
SCRIPT_DIR=$(dirname "$(realpath "$0")")
MODEL_PATH=${SCRIPT_DIR}/model/0/model.pt
DATA_PATH=${SCRIPT_DIR}/./dataset/
KEEP_PATH=${SCRIPT_DIR}/model/0/keep.npy
EPOCHS=100

# stop when error occurs
set -e

# # clean the files
# rm -f "${SCRIPT_DIR}"/*.txt
# for path in BRECQ AdaRound OBC; do
#     if [ -f "${SCRIPT_DIR}/${path}/helper.txt" ]; then
#         echo > "${SCRIPT_DIR}/${path}/helper.txt"
#     fi
#     TARGET_DIR="${SCRIPT_DIR}/${path}"
#     find "${TARGET_DIR}" -type f \( -name "*.pt" -o -name "*.pth" \) -exec rm -f {} +
# done


# train 5 target models 42
for PKEEP in 0.1 0.25 0.5 0.75 1.0; do 
    SEED=42
    # deal with the full model
    cd $SCRIPT_DIR
    if [ -f "${SCRIPT_DIR}/model/0/accu.txt" ]; then
        TRAIN_TIME=$(date)
        echo -e "\n[$TRAIN_TIME] epochs $EPOCHS pkeep $PKEEP" >> "${SCRIPT_DIR}/model/0/accu.txt"
    fi

    python3 train.py --epochs $EPOCHS --savedir ${SCRIPT_DIR}/model --seed $SEED --pkeep $PKEEP
    python3 inference.py --savedir ${SCRIPT_DIR}/model
    python3 score.py --savedir ${SCRIPT_DIR}/model

    # quantize by OBC
    cd ${SCRIPT_DIR}/OBC/
    if [ -f "${SCRIPT_DIR}/OBC/helper.txt" ]; then
        TIME=$(date)
        
        echo -e "[$TIME] pkeep $PKEEP epochs $EPOCHS \n" >> "${SCRIPT_DIR}/OBC/helper.txt"
    fi

    python main_trueobs.py resnet imagenet quant --load $MODEL_PATH --datapath $DATA_PATH \
        --seed $SEED --save W2A32.pth --batchsize 32 --wbits 2 --wasym --keep $KEEP_PATH
    python postproc.py resnet imagenet W2A32.pth  --datapath $DATA_PATH --seed $SEED --bnt --keep $KEEP_PATH


    # quantize by BRECQ
    cd ${SCRIPT_DIR}/BRECQ/
    if [ -f "${SCRIPT_DIR}/BRECQ/helper.txt" ]; then
        TIME=$(date)
        echo -e "[$TIME] pkeep $PKEEP epochs $EPOCHS \n" >> "${SCRIPT_DIR}/BRECQ/helper.txt"
    fi

    python main_imagenet.py --seed $SEED --arch resnet18 --load $MODEL_PATH --data_path $DATA_PATH \
        --save W2A32.pth --n_bits_w 2 --channel_wise --keep ${KEEP_PATH}

        
    # quantize by AdaRound
    cd ${SCRIPT_DIR}/AdaRound/
    if [ -f "${SCRIPT_DIR}/AdaRound/helper.txt" ]; then
        TIME=$(date)
        echo -e "[$TIME] pkeep $PKEEP epochs $EPOCHS \n" >> "${SCRIPT_DIR}/AdaRound/helper.txt"
    fi

    python main_quant.py --load ${MODEL_PATH} --datapath ${DATA_PATH} \
        --wbits 2 --save W2A32.pth --seed $SEED --keep ${KEEP_PATH}

done