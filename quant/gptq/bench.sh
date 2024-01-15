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
    nvidia-smi --format=csv,nounits --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.free,memory.total -lms 100 > $METHOD_EXPORTS/gpu_bench_mem_usage_$1.csv &
}

# baseline
start_mem_tracker "baseline"
python $METHOD_PATH/generation_speed.py \
--model_name_or_path $MODEL_PATH \
--tokenizer_name_or_path $MODEL_PATH \
--from_pretrained \
--dataset $DATASET_PATH \
--use_fast \
# --use_triton \
--disable_exllama 2>&1 | tee $METHOD_EXPORTS/bench_baseline.log
killall nvidia-smi

# quantize q4-128g
MODEL_Q4_128G=$METHOD_EXPORTS/q4_128g
start_mem_tracker "q4_128g"
python $METHOD_PATH/generation_speed.py \
--model_name_or_path $MODEL_Q4_128G \
--tokenizer_name_or_path $MODEL_PATH \
--dataset $DATASET_PATH \
--quantize_config_save_dir $MODEL_Q4_128G \
--disable_exllama 2>&1 | tee $METHOD_EXPORTS/bench_q4_128g.log
killall nvidia-smi

# quantize q8-128g
MODEL_Q8_128G=$METHOD_EXPORTS/q8_128g
start_mem_tracker "q8_128g"
python $METHOD_PATH/generation_speed.py \
--model_name_or_path $MODEL_Q8_128G \
--tokenizer_name_or_path $MODEL_PATH \
--dataset $DATASET_PATH \
--quantize_config_save_dir $MODEL_Q8_128G \
--disable_exllama 2>&1 | tee $METHOD_EXPORTS/bench_q8_128g.log
killall nvidia-smi