cd ../src

ARCH=(
"FE_resnet34.4-conv1-layer3_CL_plainnet34-layer4-fc_layer-classifier.layer4.0.relu2_act-classifier"
"FE_resnet34.4-conv1-layer3_CL_resnet34-layer4-fc_layer-classifier.layer4.0.relu2_act-classifier"
)

ROOT="e_receptive_field/"
OUT="mean_rf_erf-finetue"

LAYER_SIZE_ARRAY=(
2
3
5
2
)


BASE_CH=64
for LAYER_NUM in `seq 4 4`; do
    CH=$((BASE_CH * 2 ** (LAYER_NUM - 1)))
    LAYER="layer"$LAYER_NUM
    echo $LAYER
    for j in `seq 0 $((${#ARCH[@]} - 1))`; do
        echo ${ARCH[$j]}
        python analyize_rf_datas.py --root $ROOT --out $OUT --arch ${ARCH[$j]} --layer-name $LAYER --max-ch-cnt $CH --skip-kmeans --skip-gradrf
    done
done 
