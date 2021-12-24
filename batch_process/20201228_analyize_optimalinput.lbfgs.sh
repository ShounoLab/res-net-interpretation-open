cd ../src

ARCH=(
"resnet34-skip4"
"resnet34-plain4"
# "resnet34-skip"
# "resnet34-plain"
)

optim="lbfgs"
OUT="out-dir"$optim

SEEDS=(
0
)

LAYER_SIZE_ARRAY=(
2
3
5
2
)


BASE_CH=64
MAX_ITER=30
MAX_ITER=15
for i in `seq 0 $((${#SEEDS[@]} - 1))`; do
    seed=${SEEDS[$i]}
    echo $seed
    for LAYER_NUM in `seq 1 4`; do
        CH=$((BASE_CH * 2 ** (LAYER_NUM - 1)))
        LR=`awk "BEGIN {print 0.05 * $(( 2 ** (LAYER_NUM - 1) ))}"`
        result=`echo "$LR > 0.2" | bc`
        if [ $result -eq 1 ]; then
            LR=0.2
        fi
        LR=1.0
        WD=10.0
        echo $LR
        for LAYER_SIZE in `seq 0 $((LAYER_SIZE_ARRAY[LAYER_NUM - 1]))`; do
            # LAYER="layer"$LAYER_NUM"."$LAYER_SIZE".relu2"
            for j in `seq 0 $((${#ARCH[@]} - 1))`; do
                if [ ${ARCH[$j]} = "resnet34-skip4" ]; then
                    LAYER="layer"$LAYER_NUM"."$LAYER_SIZE".add_func"
                else
                    LAYER="layer"$LAYER_NUM"."$LAYER_SIZE".bn2"
                fi
                echo ${ARCH[$j]}
                echo $LAYER
                python analyize_optimalinput.py --out $OUT --arch ${ARCH[$j]} --layer-name $LAYER  --device "cuda:1" --lr $LR --mode neuron --max-iter $MAX_ITER --random uniform --seed $seed --channel-wise --batch-size 10 --wd $WD --optim $optim --zero-start
                break
            done
            break
        done
    done 
done
