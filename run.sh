# settings
SCRIPT_DIR=$(dirname "$(realpath "$0")")
MODEL_PATH=${SCRIPT_DIR}/model/0/model.pt
DATA_PATH=${SCRIPT_DIR}/../../tiny-imagenet/tiny-imagenet-200/
KEEP_PATH=${SCRIPT_DIR}/model/0/keep.npy
EPOCHS=100

# stop when error occurs
# set -e

# # clean the files
# rm -f "${SCRIPT_DIR}"/*.txt
# for path in BRECQ AdaRound OBC; do
#     if [ -f "${SCRIPT_DIR}/${path}/helper.txt" ]; then
#         echo > "${SCRIPT_DIR}/${path}/helper.txt"
#     fi
#     TARGET_DIR="${SCRIPT_DIR}/${path}"
#     find "${TARGET_DIR}" -type f \( -name "*.pt" -o -name "*.pth" \) -exec rm -f {} +
# done

PKEEP=0.5
SEED=42

# echo -e "\n[$(date)] epochs $EPOCHS pkeep $PKEEP" >> "${SCRIPT_DIR}/model/0/accu.txt"
# python3 train_model.py --epochs $EPOCHS --savedir ${SCRIPT_DIR}/model --seed $SEED --pkeep $PKEEP --dpath $DATA_PATH --dataset tiny-imagenet

# quantize by OBC
# cd ${SCRIPT_DIR}/OBC/      
# echo -e "[$(date)] pkeep $PKEEP epochs $EPOCHS \n" >> "${SCRIPT_DIR}/OBC/helper.txt"

# python main_trueobs.py resnet imagenet quant --load $MODEL_PATH --datapath $DATA_PATH \
#     --seed $SEED --save W2A32.pth --batchsize 32 --wbits 2 --wasym --keep $KEEP_PATH --data_set tiny-imagenet
# python postproc.py resnet imagenet W2A32.pth  --datapath $DATA_PATH --seed $SEED --bnt --keep $KEEP_PATH --batchsize 128 --data_set tiny-imagenet




# # train 5 target models 42
# for PKEEP in 0.25 0.5 0.75 1.0; do 
#     SEED=42
#     # deal with the full model
#     cd $SCRIPT_DIR

#     TRAIN_TIME=$(date)
#     echo -e "\n[$TRAIN_TIME] epochs $EPOCHS pkeep $PKEEP" >> "${SCRIPT_DIR}/model/0/accu.txt"

#     python3 train_model.py --epochs $EPOCHS --savedir ${SCRIPT_DIR}/model --seed $SEED --pkeep $PKEEP --dpath $DATA_PATH
#     if [ $? -ne 0 ]; then
#         echo "train_model.py failed at $(date)" >> ${SCRIPT_DIR}/log.txt
#     fi

    # quantize by OBC
    cd ${SCRIPT_DIR}/OBC/      
    echo -e "[$(date)] pkeep $PKEEP epochs $EPOCHS \n" >> "${SCRIPT_DIR}/OBC/helper.txt"


    python main_trueobs.py resnet imagenet quant --load $MODEL_PATH --datapath $DATA_PATH \
        --seed $SEED --save W2A32.pth --batchsize 32 --wbits 2 --wasym --keep $KEEP_PATH --nsamples 2048

    # python main_trueobs.py resnet imagenet quant --load $MODEL_PATH --datapath $DATA_PATH \
    #     --seed $SEED --save W3A32.pth --batchsize 32 --wbits 3 --wasym --keep $KEEP_PATH

    # python main_trueobs.py resnet imagenet quant --load $MODEL_PATH --datapath $DATA_PATH \
    #     --seed $SEED --save W4A32.pth --batchsize 32 --wbits 4 --wasym --keep $KEEP_PATH        
#     if [ $? -ne 0 ]; then
#         echo "main_trueobs.py failed at $(date)" >> ${SCRIPT_DIR}/log.txt
#     fi
    python postproc.py resnet imagenet W2A32.pth  --datapath $DATA_PATH --seed $SEED --bnt --keep $KEEP_PATH
    # python postproc.py resnet imagenet W3A32.pth  --datapath $DATA_PATH --seed $SEED --bnt --keep $KEEP_PATH
    # python postproc.py resnet imagenet W4A32.pth  --datapath $DATA_PATH --seed $SEED --bnt --keep $KEEP_PATH
#     if [ $? -ne 0 ]; then
#         echo "postproc.py failed at $(date)" >> ${SCRIPT_DIR}/log.txt
#     fi

    # quantize by BRECQ
    cd ${SCRIPT_DIR}/BRECQ/
    echo -e "[$(date)] pkeep $PKEEP epochs $EPOCHS \n" >> "${SCRIPT_DIR}/BRECQ/helper.txt"

    python main_imagenet.py --seed $SEED --arch resnet18 --load $MODEL_PATH --data_path $DATA_PATH \
        --save W2A32.pth --n_bits_w 2 --channel_wise --keep ${KEEP_PATH} --disable_8bit_head_stem --test_before_calibration
#     if [ $? -ne 0 ]; then
#         echo "main_imagenet.py failed at $(date)" >> ${SCRIPT_DIR}/log.txt
#     fi

        
    # # quantize by AdaRound
    # cd ${SCRIPT_DIR}/AdaRound/
    # echo -e "[$(date)] pkeep $PKEEP epochs $EPOCHS \n" >> "${SCRIPT_DIR}/AdaRound/helper.txt"

    # python main_quant.py --load ${MODEL_PATH} --datapath ${DATA_PATH} \
    #     --wbits 2 --save W2A32.pth --seed $SEED --keep ${KEEP_PATH} --dataset tiny-imagenet

    # python main_quant.py --load ${MODEL_PATH} --datapath ${DATA_PATH} \
    #     --wbits 3 --save W3A32.pth --seed $SEED --keep ${KEEP_PATH} --dataset tiny-imagenet

    # python main_quant.py --load ${MODEL_PATH} --datapath ${DATA_PATH} \
    #     --wbits 4 --save W4A32.pth --seed $SEED --keep ${KEEP_PATH} --dataset tiny-imagenet
#     if [ $? -ne 0 ]; then
#         echo "main_quant.py failed at $(date)" >> ${SCRIPT_DIR}/log.txt
#     fi

# done