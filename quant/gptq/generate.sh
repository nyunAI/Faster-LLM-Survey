# generates different outputs from llamacpp
METHOD=quant/gptq
METHOD_PATH=$METHOD

MODEL_NAME=Llama-2-7b-hf
DATASET_NAME=alpaca-cleaned

MODEL_PATH=model/$MODEL_NAME
DATASET_PATH=datasets/$DATASET_NAME/alpaca_data_cleaned.json

EXPORTS=exports/$MODEL_NAME
METHOD_EXPORTS=$EXPORTS/$METHOD

mkdir -p $METHOD_EXPORTS

# Function to start memory tracker
start_mem_tracker() {
    nvidia-smi --format=csv,nounits --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.free,memory.total -lms 100 > $METHOD_EXPORTS/gpu_mem_usage_$1.csv &
}

# quantize q4-128g
MODEL_Q4_128G=$METHOD_EXPORTS/q4_128g
start_mem_tracker "quantize_q4_128g"
python $METHOD_PATH/quant_with_alpaca.py \
--pretrained_model_dir $MODEL_PATH \
--quantized_model_dir $MODEL_Q4_128G \
--dataset $DATASET_PATH \
--bits 4
killall nvidia-smi

# quantize q8-128g
MODEL_Q8_128G=$METHOD_EXPORTS/q8_128g
start_mem_tracker "quantize_q8_128g"
python $METHOD_PATH/quant_with_alpaca.py \
--pretrained_model_dir $MODEL_PATH \
--quantized_model_dir $MODEL_Q8_128G \
--dataset $DATASET_PATH \
--bits 8
killall nvidia-smi

# quantize q2-128g
MODEL_Q2_128G=$METHOD_EXPORTS/q2_128g
start_mem_tracker "quantize_q2_128g"
python $METHOD_PATH/quant_with_alpaca.py \
--pretrained_model_dir $MODEL_PATH \
--quantized_model_dir $MODEL_Q2_128G \
--dataset $DATASET_PATH \
--bits 2
killall nvidia-smi

# quantize q3-128g
MODEL_Q3_128G=$METHOD_EXPORTS/q3_128g
start_mem_tracker "quantize_q3_128g"
python $METHOD_PATH/quant_with_alpaca.py \
--pretrained_model_dir $MODEL_PATH \
--quantized_model_dir $MODEL_Q3_128G \
--dataset $DATASET_PATH \
--bits 3
killall nvidia-smi

# # quantize triton q4-128g
# MODEL_TRT_Q4_128G=$METHOD_EXPORTS/trt_q4_128g
# start_mem_tracker "quantize_trt_q4_128g"
# python $METHOD_PATH/quant_with_alpaca.py \
# --pretrained_model_dir $MODEL_PATH \
# --quantized_model_dir $MODEL_Q4_128G \
# --dataset $DATASET_PATH \
# --bits 4 \
# --use_triton
# killall nvidia-smi
