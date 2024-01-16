# runs inside the tensorrtllm docker container

BASE=$(pwd)
METHOD=engine/tensorrtllm
METHOD_PATH=$METHOD/tensorrtllm
METHOD_SUBPATH=/app/tensorrt_llm/examples/llama

MODEL_NAME=Llama-2-7b-hf
MODEL_PATH=model/$MODEL_NAME
EXPORTS=exports/$MODEL_NAME
METHOD_EXPORTS=$EXPORTS/$METHOD

DATASET_NAME=alpaca-cleaned
DATASET_PATH=datasets/$DATASET_NAME

mkdir -p $METHOD_EXPORTS

apt-get install psmisc

# Function to start memory tracker
start_mem_tracker() {
    nvidia-smi --format=csv,nounits --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.free,memory.total -lms 100 > $METHOD_EXPORTS/gpu_bench_mem_usage_$1.csv &
}

start_mem_tracker "baseline"
python $METHOD_PATH/benchmarks/python/benchmark.py \
-m llama_7b \
--mode plugin \
--batch_size "1;8;64" \
--input_output_len "60,20;128,20" 2>&1 | tee $METHOD_EXPORTS/benchmark_baseline.log
killall nvidia-smi