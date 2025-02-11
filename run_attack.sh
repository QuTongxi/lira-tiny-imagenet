# settings
SCRIPT_DIR=$(dirname "$(realpath "$0")")
MODEL_PATH=${SCRIPT_DIR}/model/0/model.pt
DATA_PATH=${SCRIPT_DIR}/../../tiny-imagenet/tiny-imagenet-200/
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

# # train and handle shadow models
# for shadow_id in {0..15}; do
#     python3 train_model.py --epochs $EPOCHS --shadow_id $shadow_id --seed $RANDOM --n_shadows 16 --pkeep 0.5 --dpath $DATA_PATH --dataset tiny-imagenet --savedir exp/tiny-imageNet
# done
python3 inference.py --savedir exp/tiny-imageNet --dpath $DATA_PATH
python3 score.py --savedir exp/tiny-imageNet --dpath $DATA_PATH

SEED=42
# train 5 target models
# for SEED in 42 83 57; do
    # # deal with the full model
    # TIME=$(date)

    # echo -e "[$TIME] epochs $EPOCHS \n" >> "${SCRIPT_DIR}/model/0/accu.txt"
    # python3 train_model.py --epochs $EPOCHS --savedir ${SCRIPT_DIR}/model --seed $SEED --pkeep 0.5 --dpath $DATA_PATH --dataset tiny-imagenet
    # python3 inference.py --savedir ${SCRIPT_DIR}/model --dpath $DATA_PATH
    # python3 score.py --savedir ${SCRIPT_DIR}/model --dpath $DATA_PATH

    # # for path in BRECQ AdaRound OBC; do
    # if [ -f "${SCRIPT_DIR}/${path}/helper.txt" ]; then       
    #         echo -e "[$TIME] epochs $EPOCHS \n" >> "${SCRIPT_DIR}/${path}/helper.txt"
    # fi

    # quantize by OBC
    cd ${SCRIPT_DIR}/OBC/
    for idx in {2,3,4};do
        python main_trueobs.py resnet imagenet quant --load $MODEL_PATH --datapath $DATA_PATH \
            --seed $SEED --save W${idx}A32.pth --batchsize 32 --wbits ${idx} --wasym --keep $KEEP_PATH
        python postproc.py resnet imagenet W${idx}A32.pth  --datapath $DATA_PATH --seed $SEED --bnt --keep $KEEP_PATH
    done

    # quantize by BRECQ
    cd ${SCRIPT_DIR}/BRECQ/
    for idx in {2,3,4};do
        python main_imagenet.py --seed $SEED --arch resnet18 --load $MODEL_PATH --data_path $DATA_PATH \
            --save W${idx}A32.pth --n_bits_w ${idx} --channel_wise --keep ${KEEP_PATH}
    done
        
    # quantize by AdaRound
    cd ${SCRIPT_DIR}/AdaRound/
    for idx in {2,3,4};do
        python main_quant.py --load ${MODEL_PATH} --datapath ${DATA_PATH} \
            --wbits 2 --save W4A32.pth --seed $SEED --keep ${KEEP_PATH} --dataset tiny-imagenet
    done

    # score the quantized models
    cd ${SCRIPT_DIR}/
    python3 score.py --savedir dat/AdaRound/ --dpath $DATA_PATH
    python3 score.py --savedir dat/BRECQ/ --dpath $DATA_PATH
    python3 score.py --savedir dat/OBC/ --dpath $DATA_PATH

    # apply the attack to models
    if [ -f "${SCRIPT_DIR}/out_data.txt" ];then
        echo -e "\n[$TIME] epochs $EPOCHS \n" >> "${SCRIPT_DIR}/out_data.txt"
    fi

    python3 plot.py --keep $KEEP_PATH --scores ${SCRIPT_DIR}/model/0/scores.npy --name FULL --savedir exp/tiny-imageNet
    find ${SCRIPT_DIR}/dat -type f -name "scores.npy" | while read -r file; do
        parent_dir=$(basename "$(dirname "$file")") 
        grandparent_dir=$(basename "$(dirname "$(dirname "$file")")")

        name="${grandparent_dir}_${parent_dir}"

        python3 plot.py --keep $KEEP_PATH --scores $file --name $name --savedir exp/tiny-imageNet 
    done
# done
