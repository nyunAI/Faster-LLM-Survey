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

# copy the tokenizer model and files to quantized model paths

# baseline
start_mem_tracker "baseline_exllamav2"
python $METHOD_PATH/generation_speed.py \
--model_name_or_path $MODEL_PATH \
--tokenizer_name_or_path $MODEL_PATH \
--from_pretrained \
--dataset $DATASET_PATH \
--num_beams 4 \
--per_gpu_max_memory 40 \
--cpu_max_memory 0 \
--use_fast \
--max_new_tokens 128 \
--disable_exllama 2>&1 | tee $METHOD_EXPORTS/bench_exllamav2_baseline.log
killall nvidia-smi

# baseline
start_mem_tracker "baseline_exllama"
python $METHOD_PATH/generation_speed.py \
--model_name_or_path $MODEL_PATH \
--tokenizer_name_or_path $MODEL_PATH \
--from_pretrained \
--dataset $DATASET_PATH \
--num_beams 4 \
--per_gpu_max_memory 40 \
--use_fast \
--max_new_tokens 128 \
--cpu_max_memory 0 \
--use_fast 2>&1 | tee $METHOD_EXPORTS/bench_exllama_baseline.log
killall nvidia-smi

# quantize q4-128g
MODEL_Q4_128G=$METHOD_EXPORTS/q4_128g
start_mem_tracker "q4_128g_exllamav2"
python $METHOD_PATH/generation_speed.py \
--model_name_or_path $MODEL_Q4_128G \
--tokenizer_name_or_path $MODEL_PATH \
--dataset $DATASET_PATH \
--quantize_config_save_dir $MODEL_Q4_128G \
--num_beams 4 \
--per_gpu_max_memory 40 \
--cpu_max_memory 0 \
--use_fast \
--max_new_tokens 128 \
--disable_exllama 2>&1 | tee $METHOD_EXPORTS/bench_exllamav2_q4_128g.log
killall nvidia-smi

# quantize q4-128g
MODEL_Q4_128G=$METHOD_EXPORTS/q4_128g
start_mem_tracker "q4_128g_exllama"
python $METHOD_PATH/generation_speed.py \
--model_name_or_path $MODEL_Q4_128G \
--tokenizer_name_or_path $MODEL_PATH \
--dataset $DATASET_PATH \
--quantize_config_save_dir $MODEL_Q4_128G \
--num_beams 4 \
--per_gpu_max_memory 40 \
--use_fast \
--max_new_tokens 128 \
--cpu_max_memory 0 \
--use_fast 2>&1 | tee $METHOD_EXPORTS/bench_exllama_q4_128g.log
killall nvidia-smi

# quantize q8-128g
MODEL_Q8_128G=$METHOD_EXPORTS/q8_128g
start_mem_tracker "q8_128g_exllamav2"
python $METHOD_PATH/generation_speed.py \
--model_name_or_path $MODEL_Q8_128G \
--tokenizer_name_or_path $MODEL_PATH \
--dataset $DATASET_PATH \
--quantize_config_save_dir $MODEL_Q8_128G \
--num_beams 4 \
--per_gpu_max_memory 40 \
--cpu_max_memory 0 \
--use_fast \
--max_new_tokens 128 \
--disable_exllama 2>&1 | tee $METHOD_EXPORTS/bench_exllamav2_q8_128g.log
killall nvidia-smi

# quantize q8-128g
MODEL_Q8_128G=$METHOD_EXPORTS/q8_128g
start_mem_tracker "q8_128g_exllama"
python $METHOD_PATH/generation_speed.py \
--model_name_or_path $MODEL_Q8_128G \
--tokenizer_name_or_path $MODEL_PATH \
--dataset $DATASET_PATH \
--quantize_config_save_dir $MODEL_Q8_128G \
--num_beams 4 \
--per_gpu_max_memory 40 \
--use_fast \
--max_new_tokens 128 \
--cpu_max_memory 0 \
--use_fast 2>&1 | tee $METHOD_EXPORTS/bench_exllama_q8_128g.log
killall nvidia-smi