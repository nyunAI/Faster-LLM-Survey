METHOD=engine/vllm
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

# List of model types
# gptq
MODEL_Q4_128G=gptq/q4_128g
MODEL_Q8_128G=gptq/q8_128g # 8bit quantization is not supported
# awq
MODEL_AWQ_Q4_GEMM=awq/gemm
# TODO: Add squeezellm

# baseline
start_mem_tracker "baseline"
python $METHOD_PATH/benchmark_throughput.py --backend vllm --dataset $DATASET_PATH --model $MODEL_PATH --tokenizer $MODEL_PATH --num-prompts=1000 2>&1 | tee $METHOD_EXPORTS/bench_baseline.log
killall nvidia-smi

# Loop through each model type
for MODEL_TYPE in "$MODEL_Q4_128G" "$MODEL_AWQ_Q4_GEMM"; do
    QUANT_MODEL_PATH=$EXPORTS/quant/$MODEL_TYPE

    QUANT_TYPE=$(echo $MODEL_TYPE | cut -d'/' -f1)
    MODEL_STR=$(echo $MODEL_TYPE | sed 's/\//_/g')

    start_mem_tracker $MODEL_STR

    # Run llama-bench for the current model type
    python $METHOD_PATH/benchmark_throughput.py --backend vllm --dataset $DATASET_PATH --quantization $QUANT_TYPE --model $QUANT_MODEL_PATH --tokenizer $MODEL_PATH --num-prompts=1000 2>&1 | tee $METHOD_EXPORTS/bench_$MODEL_STR.log

    # Stop memory tracker for the current model type
    killall nvidia-smi
done