cd ../src

ARCH=(
"model-path0"
"model-path1"
)

ROOT="finetune-epoch90"
OUT=/data2/genta/resnet/analysis

BASE_CH=64
# LAYER=4
# CH=$((BASE_CH * 2 ** (LAYER - 1)))
# j=0
# CH=77
# echo ${ARCH[$j]} ${LAYER} $CH $START_CH $END_CH
# python analyize_preact.py -l layer$LAYER -r $ROOT -o $OUT -a ${ARCH[$j]} --wandb-flag --start-end-channel 76 78
for j in `seq 0 $((${#ARCH[@]} - 1))`; do
    echo ${ARCH[$j]}
    for LAYER in `seq 4 4`
    do
        CH=$((BASE_CH * 2 ** (LAYER - 1)))
        echo ${LAYER} $CH
        python analyize_preact.py -l layer$LAYER -r $ROOT -o $OUT -a ${ARCH[$j]} --wandb-flag --start-end-channel $((77 + 83)) 512

    done
done

