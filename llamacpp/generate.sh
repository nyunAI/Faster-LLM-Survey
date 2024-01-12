# generates different outputs from llamacpp
METHOD=llamacpp
METHOD_PATH=$METHOD/llama.cpp

MODEL_NAME=Llama-2-7b-hf
MODEL_PATH=model/$MODEL_NAME
EXPORTS=$MODEL_PATH/exports
METHOD_EXPORTS=$EXPORTS/$METHOD

mkdir -p $METHOD_EXPORTS

# Function to start memory tracker
start_mem_tracker() {
    nvidia-smi --format=csv,nounits --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.free,memory.total -lms 100 > $METHOD_EXPORTS/gpu_mem_usage_$1.csv &
}

# # 
# start_cpu_mem_tracker() {
#     (echo "free_memory,swap_in,swap_out,io_block,inactive,active,si,so"; vmstat 1 | awk 'NR > 2 {print $4","$5","$6","$7","$8","$9","$10}') > $METHOD_EXPORTS/cpu_mem_usage_$1.csv &
# }

# gguf-f16
MODEL_GGUF=$METHOD_EXPORTS/ggml-model-f16.gguf
start_mem_tracker "convert"
python $METHOD_PATH/convert.py $MODEL_PATH --outfile $MODEL_GGUF
killall nvidia-smi

# quantized q4_0
MODEL_Q4_0_GGUF=$METHOD_EXPORTS/ggml-model-q4_0.gguf
start_mem_tracker "quantize_q4_0"
$METHOD_PATH/quantize $MODEL_GGUF $MODEL_Q4_0_GGUF q4_0
killall nvidia-smi

# quantized q4_k_m
MODEL_Q4_K_M_GGUF=$METHOD_EXPORTS/ggml-model-q4_k_m.gguf
start_mem_tracker "quantize_q4_k_m"
$METHOD_PATH/quantize $MODEL_GGUF $MODEL_Q4_K_M_GGUF q4_k_m
killall nvidia-smi
