# generates different outputs from llamacpp
METHOD=engine/llamacpp
METHOD_PATH=$METHOD/llama.cpp

MODEL_NAME=Llama-2-7b-hf
MODEL_PATH=model/$MODEL_NAME
EXPORTS=exports/$MODEL_NAME
METHOD_EXPORTS=$EXPORTS/$METHOD

AWQ_CACHE_PATH=model/awq_cache

GPTQ_EXPORTS=$EXPORTS/quant/gptq
HF_EXPORTS=$EXPORTS/quant/hf

mkdir -p $METHOD_EXPORTS

# Function to start memory tracker
start_mem_tracker() {
    nvidia-smi --format=csv,nounits --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.free,memory.total -lms 100 > $METHOD_EXPORTS/gpu_mem_usage_$1.csv &
}

# # 
# start_cpu_mem_tracker() {
#     (echo "free_memory,swap_in,swap_out,io_block,inactive,active,si,so"; vmstat 1 | awk 'NR > 2 {print $4","$5","$6","$7","$8","$9","$10}') > $METHOD_EXPORTS/cpu_mem_usage_$1.csv &
# }

# # gguf-f16
MODEL_GGUF=$METHOD_EXPORTS/ggml-model-f16.gguf
start_mem_tracker "convert"
python $METHOD_PATH/convert.py $MODEL_PATH --outfile $MODEL_GGUF
killall nvidia-smi

# quantized q4_0
MODEL_Q4_0_GGUF=$METHOD_EXPORTS/ggml-model-q4_0.gguf
start_mem_tracker "quantize_q4_0"
$METHOD_PATH/quantize $MODEL_GGUF $MODEL_Q4_0_GGUF q4_0
killall nvidia-smi

# quantized q8_0
MODEL_Q8_0_GGUF=$METHOD_EXPORTS/ggml-model-q8_0.gguf
start_mem_tracker "quantize_q8_0"
$METHOD_PATH/quantize $MODEL_GGUF $MODEL_Q8_0_GGUF q8_0
killall nvidia-smi

quantized q2_k
MODEL_Q2_K_GGUF=$METHOD_EXPORTS/ggml-model-q2_k.gguf
start_mem_tracker "quantize_q2_k"
$METHOD_PATH/quantize $MODEL_GGUF $MODEL_Q2_K_GGUF q2_k
killall nvidia-smi

# quantized q4_k_m
MODEL_Q4_K_M_GGUF=$METHOD_EXPORTS/ggml-model-q4_k_m.gguf
start_mem_tracker "quantize_q4_k_m"
$METHOD_PATH/quantize $MODEL_GGUF $MODEL_Q4_K_M_GGUF q4_k_m
killall nvidia-smi

# quantized q4_k_s
MODEL_Q4_K_S_GGUF=$METHOD_EXPORTS/ggml-model-q4_k_s.gguf
start_mem_tracker "quantize_q4_k_s"
$METHOD_PATH/quantize $MODEL_GGUF $MODEL_Q4_K_S_GGUF q4_k_s
killall nvidia-smi



# change the convert.py file  to
# L1558:        sys.path.append(str(Path(__file__).resolve().parent / "awq-py" / "awq"))
# L1559         from apply_awq import add_scale_weights


# awq
MODEL_AWQ_W4_CACHE_PATH=$AWQ_CACHE_PATH/llama-2-7b-w4-g128.pt

# awq4-gguf-f16
MODEL_AWQ_GGUF=$METHOD_EXPORTS/ggml-model-f32-awq4g128.gguf
start_mem_tracker "convert_awq"
python $METHOD_PATH/convert.py $MODEL_PATH --outfile $MODEL_AWQ_GGUF --awq-path $MODEL_AWQ_W4_CACHE_PATH
killall nvidia-smi

# quantized awq_q4_0
MODEL_AWQ_Q4_0_GGUF=$METHOD_EXPORTS/ggml-model-awq_q4_0.gguf
start_mem_tracker "quantize_awq_q4_0"
$METHOD_PATH/quantize $MODEL_AWQ_GGUF $MODEL_AWQ_Q4_0_GGUF q4_0
killall nvidia-smi

