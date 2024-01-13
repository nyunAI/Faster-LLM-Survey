METHOD=engine/exllama
METHOD_PATH=$METHOD/exllama

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
python $METHOD_PATH/test_benchmark_inference.py \
-d $MODEL_PATH \
-p -ppl 2>&1 | tee $METHOD_EXPORTS/bench_baseline.log
killall nvidia-smi

# quantized awq-gemm
MODEL_AWQ_GEMM=$EXPORTS/quant/awq/gemm
start_mem_tracker "quantized_awq_gemm"
python $METHOD_PATH/test_benchmark_inference.py \
-d $MODEL_AWQ_GEMM \
-p -ppl 2>&1 | tee $METHOD_EXPORTS/quantized_awq_gemm.log
killall nvidia-smi