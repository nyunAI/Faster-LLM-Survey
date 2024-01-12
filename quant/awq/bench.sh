# generates different outputs from llamacpp
METHOD=quant/awq
METHOD_PATH=$METHOD

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
python $METHOD_PATH/benchmark.py \
--model $MODEL_PATH \
--pretrained \
--batch_size 1 2>&1 | tee $METHOD_EXPORTS/bench_baseline.log
killall nvidia-smi

# quantize awq-gemm
MODEL_AWQ_GEMM=$METHOD_EXPORTS/gemm
start_mem_tracker "quantize_awq_gemm"
python $METHOD_PATH/benchmark.py \
--model $MODEL_AWQ_GEMM \
--batch_size 1 2>&1 | tee $METHOD_EXPORTS/bench_quantize_awq_gemm.log
killall nvidia-smi

# quantize awq-gemv
MODEL_AWQ_GEMV=$METHOD_EXPORTS/gemv
start_mem_tracker "quantize_awq_gemv"
python $METHOD_PATH/benchmark.py \
--model $MODEL_AWQ_GEMV \
--batch_size 1 2>&1 | tee $METHOD_EXPORTS/bench_quantize_awq_gemv.log
killall nvidia-smi
