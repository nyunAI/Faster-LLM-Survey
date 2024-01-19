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

apt-get update
apt-get install psmisc

# Function to start memory tracker
start_mem_tracker() {
    nvidia-smi --format=csv,nounits --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.free,memory.total -lms 100 > $METHOD_EXPORTS/gpu_bench_mem_usage_$1.csv &
}

# start_mem_tracker "baseline"
# sleep 3
# python $METHOD_PATH/benchmarks/python/benchmark.py \
# -m llama_7b \
# --mode plugin \
# --batch_size "1" \
# --input_output_len "128,128" 2>&1 | tee $METHOD_EXPORTS/benchmark_baseline.log
# killall nvidia-smi
# sleep 3


BASE_F16=base_f16
MODEL_INT8=int8
MODEL_INT4_AWQ=int4_awq
MODEL_SQ0_8=sq0_8
MODEL_GPTQ=gptq


for MODEL_TYPE in "$BASE_F16" "$MODEL_INT8" "$MODEL_INT4_AWQ" "$MODEL_SQ0_8" "$MODEL_GPTQ" ; do
    QUANT_MODEL_ENGINE_PATH=$METHOD_EXPORTS/$MODEL_TYPE/engine

    QUANT_TYPE=$(echo $MODEL_TYPE | cut -d'/' -f1)
    MODEL_STR=$(echo $MODEL_TYPE | sed 's/\//_/g')

    start_mem_tracker $QUANT_TYPE
    sleep 3
    python $METHOD_PATH/benchmarks/python/benchmark.py \
        -m llama_7b \
        --engine_dir $QUANT_MODEL_ENGINE_PATH \
        --mode plugin \
        --batch_size "1" \
        --input_output_len "128,128" 2>&1 | tee $METHOD_EXPORTS/benchmark_$MODEL_TYPE.log
    killall nvidia-smi
    sleep 3
done



