# Description: Generate the quantization results for all methods
METHOD=quant/hf
METHOD_PATH=$METHOD

MODEL_NAME=Llama-2-7b-hf
DATASET_NAME=alpaca-cleaned

MODEL_PATH=model/$MODEL_NAME
DATASET_PATH=datasets/$DATASET_NAME/alpaca_data_cleaned.json

EXPORTS=exports_2/$MODEL_NAME
METHOD_EXPORTS=$EXPORTS/$METHOD

mkdir -p $METHOD_EXPORTS


cd $METHOD_PATH

for i in {4..7}; do
    python generate.py $i
done