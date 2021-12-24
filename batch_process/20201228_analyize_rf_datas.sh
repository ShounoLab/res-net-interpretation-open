cd ../src

ARCH=(
"20200409_resnet34"
"20200409_plainnet34"
"resnet34-skip_"
"resnet34-plain_"
)

ROOT="root-dir"
OUT="mean_rf_erf"

LAYER_SIZE_ARRAY=(
2
3
5
2
)


BASE_CH=64
for LAYER_NUM in `seq 1 4`; do
    CH=$((BASE_CH * 2 ** (LAYER_NUM - 1)))
    LAYER="layer"$LAYER_NUM
    echo $LAYER
    for j in `seq 0 $((${#ARCH[@]} - 1))`; do
        echo ${ARCH[$j]}
        python analyize_rf_datas.py --root $ROOT --out $OUT --arch ${ARCH[$j]} --layer-name $LAYER --max-ch-cnt $CH --skip-kmeans --skip-gradrf
    done
done 
