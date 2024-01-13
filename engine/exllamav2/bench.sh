METHOD=engine/exllamav2
METHOD_PATH=$METHOD/exllamav2

MODEL_NAME=Llama-2-7b-hf
MODEL_PATH=model/$MODEL_NAME
EXPORTS=exports/$MODEL_NAME
METHOD_EXPORTS=$EXPORTS/$METHOD

mkdir -p $METHOD_EXPORTS

# Function to start memory tracker
start_mem_tracker() {
    nvidia-smi --format=csv,nounits --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.free,memory.total -lms 100 > $METHOD_EXPORTS/gpu_bench_mem_usage_$1.csv &
}

# baseline
start_mem_tracker "baseline"
python $METHOD_PATH/test_inference.py -m $MODEL_PATH -ps -s 2>&1 | tee $METHOD_EXPORTS/bench_baseline.log
killall nvidia-smi

# List of model types
MODEL_4BPW_EXL2="4bpw-exl2"
MODEL_6BPW_EXL2="6bpw-exl2"

# Loop through each model type
for MODEL_TYPE in "$MODEL_4BPW_EXL2" "$MODEL_6BPW_EXL2"; do
    MODEL_PATH=$METHOD_EXPORTS/$MODEL_TYPE

    start_mem_tracker $MODEL_TYPE

    # Run llama-bench for the current model type
    python $METHOD_PATH/test_inference.py -m $MODEL_PATH -ps -s 2>&1 | tee $METHOD_EXPORTS/bench_$MODEL_TYPE.log

    # Stop memory tracker for the current model type
    killall nvidia-smi
done