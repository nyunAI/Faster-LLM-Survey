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
    nvidia-smi --format=csv,nounits --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.free,memory.total -lms 100 > $METHOD_EXPORTS/gpu_mem_usage_$1.csv &
}

# quantize awq-gemm
MODEL_AWQ_GEMM=$METHOD_EXPORTS/w4_gemm
start_mem_tracker "quantize_w4_gemm"
python $METHOD_PATH/quantize.py \
--model_path $MODEL_PATH \
--quant_path $MODEL_AWQ_GEMM \
--zero_point \
--q_group_size 128 \
--w_bit 4 \
--version "GEMM"
killall nvidia-smi

# quantize awq-gemv
MODEL_AWQ_GEMV=$METHOD_EXPORTS/w4_gemv
start_mem_tracker "quantize_w4_gemv"
python $METHOD_PATH/quantize.py \
--model_path $MODEL_PATH \
--quant_path $MODEL_AWQ_GEMV \
--zero_point \
--q_group_size 128 \
--w_bit 4 \
--version "GEMV"
killall nvidia-smi