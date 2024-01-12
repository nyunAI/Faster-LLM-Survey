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
    nvidia-smi --format=csv,nounits --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.free,memory.total -lms 100 > $METHOD_EXPORTS/gpu_bench_mem_usage_$1.csv &
}

# List of model types
MODEL_GGML_F16="ggml-model-f16"
MODEL_GGML_Q4_0="ggml-model-q4_0"
MODEL_GGML_Q4_K_M="ggml-model-q4_k_m"

# Loop through each model type
for MODEL_TYPE in "$MODEL_GGML_F16" "$MODEL_GGML_Q4_0" "$MODEL_GGML_Q4_K_M"; do
    MODEL_PATH=$METHOD_EXPORTS/$MODEL_TYPE.gguf

    start_mem_tracker $MODEL_TYPE

    # Run llama-bench for the current model type
    $METHOD_PATH/llama-bench -m $MODEL_PATH -p 3968 -n 128 2>&1 | tee $METHOD_EXPORTS/bench_$MODEL_TYPE.log

    # Stop memory tracker for the current model type
    killall nvidia-smi
done