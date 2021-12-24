cd ../src

ARCH=(
"resnet34-skip4"
# "resnet34-plain4"
# "resnet34-skip"
# "resnet34-plain"
)

OUT="optimalinput-other"

LAYER_SIZE_ARRAY=(
2
3
5
2
)


BASE_CH=64
for LAYER_NUM in `seq 1 4`; do
    CH=$((BASE_CH * 2 ** (LAYER_NUM - 1)))
    LR=`awk "BEGIN {print 0.05 * $(( 2 ** (LAYER_NUM - 1) ))}"`
    result=`echo "$LR > 0.2" | bc`
    if [ $result -eq 1 ]; then
        LR=0.2
    fi
    echo $LR
    for LAYER_SIZE in `seq 0 $((LAYER_SIZE_ARRAY[LAYER_NUM - 1]))`; do
        LAYER="layer"$LAYER_NUM"."$LAYER_SIZE".relu2"
        echo $LAYER
        for j in `seq 0 $((${#ARCH[@]} - 1))`; do
            echo ${ARCH[$j]}
            python analyize_optimalinput.py --out $OUT --arch ${ARCH[$j]} --layer-name $LAYER --mode neuron --device "cuda:1" --lr $LR --mode other --max-iter 10 --zero-start
        done
    done
done 
