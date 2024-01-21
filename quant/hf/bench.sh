BASE=$(pwd)
# Description: Generate the quantization results for all methods
METHOD=quant/hf
METHOD_PATH=$METHOD

MODEL_NAME=Llama-2-7b-hf
DATASET_NAME=alpaca-cleaned

MODEL_PATH=model/$MODEL_NAME
DATASET_PATH=datasets/$DATASET_NAME/alpaca_data_cleaned.json

EXPORTS=exports/$MODEL_NAME
METHOD_EXPORTS=$EXPORTS/$METHOD



cd $METHOD_PATH

# for i in {0..7}; do
#     python bench.py $i 2>&1 | tee $BASE/$METHOD_EXPORTS/bench_$i.log
# done

python bench.py "baseline" 2>&1 | tee $BASE/$METHOD_EXPORTS/bench_baseline.log