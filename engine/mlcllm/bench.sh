
METHOD=engine/mlcllm
METHOD_PATH=$METHOD/mlcllm

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
# MODEL_Q0F16_1_MLC=$METHOD_EXPORTS/q0f16

# quantize q4f16_1
MODEL_Q4F16_1_MLC=$METHOD_EXPORTS/q4f16_1
start_mem_tracker "q4f16_1"
mlc_chat bench $MODEL_Q4F16_1_MLC \
--device "cuda:0" \
--prompt "What is the meaning of life?" \
--generate-length 256 \
--model-lib-path $MODEL_Q4F16_1_MLC/$MODEL_NAME-q4f16_1-cuda.so 2>&1 | tee $METHOD_EXPORTS/bench_q4f16_1.log
killall nvidia-smi