cd ../src

ARCH=(
"resnet34-skip4"
"resnet34-plain4"
# "resnet34-skip"
# "resnet34-plain"
)

optim="lbfgs"

SEEDS=(
0
# 12312020
# 20201231
)

LAYER_SIZE_ARRAY=(
2
3
5
2
)

hoge_flag=false

LRs=(
1.0
0.9
)

WDs=(
1.0
50000000.0
)

BASE_CH=64
MAX_ITER=15
for i_lr in `seq 0 $((${#LRs[@]} - 1 ))`; do
    LR=${LRs[${i_lr}]}
for i_wd in `seq 0 $((${#WDs[@]} - 1 ))`; do
    WD=${WDs[${i_wd}]}
for i in `seq 0 $((${#SEEDS[@]} - 1))`; do
    seed=${SEEDS[$i]}
    echo $seed
    for LAYER_NUM in `seq 1 4`; do
        CH=$((BASE_CH * 2 ** (LAYER_NUM - 1)))
        if "${hoge_flag}"; then
            if [ $LAYER_NUM -eq 4 ]; then
                LR=0.9
            else
                LR=1.0
            fi
            WD=1.0
        fi
        echo $LR $WD
        for LAYER_SIZE in `seq 0 $((LAYER_SIZE_ARRAY[LAYER_NUM - 1]))`; do
            for j in `seq 0 $((${#ARCH[@]} - 1))`; do
                if [ ${ARCH[$j]} = "resnet34-skip4" ]; then
                    LAYER="layer"$LAYER_NUM"."$LAYER_SIZE".add_func"
                    if "${hoge_flag}"; then
                    if [ $LAYER_NUM -eq 3 -a $LAYER_SIZE -gt 3 ]; then
                        WD=50000000.0
                        echo $LR $WD
                    fi
                    fi
                else
                    LAYER="layer"$LAYER_NUM"."$LAYER_SIZE".bn2"
                fi
                echo ${ARCH[$j]}
                echo $LAYER
                OUT="/data2/genta/resnet/analysis/optimalinput-"$optim"-lr"$LR"-wd"$WD
                python analyize_optimalinput.py --out $OUT --arch ${ARCH[$j]} --layer-name $LAYER  --device "cuda:1" --lr $LR --mode neuron --max-iter $MAX_ITER  --seed $seed  --batch-size 256 --wd $WD --optim $optim --zero-start --off-sorted-channel
            done
        done
    done 
done
done
done
