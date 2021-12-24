cd ../src

ARCH_RES=("resnet34-skip" "resnet34-skip2" "resnet34-skip3")
ARCH_PLAIN=("resnet34-plain" "resnet34-plain2" "resnet34-plain3")

OUT="out-dir"

VAL_LIST="${PWD}/random_sample_val.txt"
MAX_ITER=100

for j in `seq 0 $((${#ARCH_RES[@]} - 1))`; do
    echo ${ARCH_RES[$j]} ${ARCH_PLAIN[$j]}
    LAYER="maxpool"
    python make_rf_datas.py --max-iter $MAX_ITER  --val-list $VAL_LIST --out $OUT --arch ${ARCH_RES[$j]} --layer-name $LAYER
    python make_rf_datas.py --max-iter $MAX_ITER  --val-list $VAL_LIST --out $OUT --arch ${ARCH_PLAIN[$j]} --layer-name $LAYER
    
    for i in `seq 0 2`
    do
        LAYER="layer1."$i".relu2"
        python make_rf_datas.py --max-iter $MAX_ITER  --val-list $VAL_LIST --out $OUT --arch ${ARCH_RES[$j]} --layer-name $LAYER
        python make_rf_datas.py --max-iter $MAX_ITER  --val-list $VAL_LIST --out $OUT --arch ${ARCH_PLAIN[$j]} --layer-name $LAYER
    done
    
     for i in `seq 0 3`
     do
         LAYER="layer2."$i".relu2"
         python make_rf_datas.py --max-iter $MAX_ITER  --val-list $VAL_LIST --out $OUT --arch ${ARCH_RES[$j]} --layer-name $LAYER
         python make_rf_datas.py --max-iter $MAX_ITER  --val-list $VAL_LIST --out $OUT --arch ${ARCH_PLAIN[$j]} --layer-name $LAYER
     done
   
     for i in `seq 0 5`
     do
         LAYER="layer3."$i".relu2"
         python make_rf_datas.py --max-iter $MAX_ITER  --val-list $VAL_LIST --out $OUT --arch ${ARCH_RES[$j]} --layer-name $LAYER
         python make_rf_datas.py --max-iter $MAX_ITER  --val-list $VAL_LIST --out $OUT --arch ${ARCH_PLAIN[$j]} --layer-name $LAYER
     done

     for i in `seq 0 2`
     do
         LAYER="layer4."$i".relu2"
         echo $LAYER
         python make_rf_datas.py --max-iter $MAX_ITER  --val-list $VAL_LIST --out $OUT --arch ${ARCH_RES[$j]} --layer-name $LAYER
         python make_rf_datas.py --max-iter $MAX_ITER  --val-list $VAL_LIST --out $OUT --arch ${ARCH_PLAIN[$j]} --layer-name $LAYER
     done
done 
