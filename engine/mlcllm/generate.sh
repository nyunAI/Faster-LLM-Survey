METHOD=engine/mlcllm
METHOD_PATH=$METHOD/mlcllm

MODEL_NAME=Llama-2-7b-hf

MODEL_PATH=model/$MODEL_NAME

EXPORTS=exports/$MODEL_NAME
METHOD_EXPORTS=$EXPORTS/$METHOD

OMNIQUANT_TEMP=$METHOD_EXPORTS/temp

mkdir -p $METHOD_EXPORTS
# mkdir -p $OMNIQUANT_TEMP

# Function to start memory tracker
start_mem_tracker() {
    nvidia-smi --format=csv,nounits --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.free,memory.total -lms 100 > $METHOD_EXPORTS/gpu_mem_usage_$1.csv &
}

AWQ_EXPORTS=$EXPORTS/quant/awq

# generate quantized model
1. Convert weight q4f16_1
MODEL_Q4F16_1_MLC=$METHOD_EXPORTS/q4f16_1
start_mem_tracker "quantize_q4f16_1"
mlc_chat convert_weight $MODEL_PATH \
    --quantization q4f16_1 \
    -o $MODEL_Q4F16_1_MLC
# 2. gen_config: generate mlc-chat-config.json and process tokenizers
mlc_chat gen_config $MODEL_PATH \
    --quantization q4f16_1 --conv-template "llama-2" \
    -o $MODEL_Q4F16_1_MLC
# 3. compile: compile model library with specification in mlc-chat-config.json
mlc_chat compile $MODEL_Q4F16_1_MLC/mlc-chat-config.json \
    --device cuda -o $MODEL_Q4F16_1_MLC/$MODEL_NAME-q4f16_1-cuda.so
killall nvidia-smi

# 1. Convert weight q4f16_awq_gemm
MODEL_Q4F16_AWQ_MLC_GEMM=$METHOD_EXPORTS/q4f16_autoawq_gemm
start_mem_tracker "quantize_q4f16_autoawq_gemm"
mlc_chat convert_weight $MODEL_PATH \
    --quantization q4f16_autoawq \
    --source-format awq \
    --source $AWQ_EXPORTS/gemm/model.safetensors \
    -o $MODEL_Q4F16_AWQ_MLC_GEMM
# 2. gen_config: generate mlc-chat-config.json and process tokenizers
mlc_chat gen_config $MODEL_PATH \
    --quantization q4f16_autoawq --conv-template "llama-2" \
    -o $MODEL_Q4F16_AWQ_MLC_GEMM
# 3. compile: compile model library with specification in mlc-chat-config.json
mlc_chat compile $MODEL_Q4F16_AWQ_MLC_GEMM/mlc-chat-config.json \
    --device cuda -o $MODEL_Q4F16_AWQ_MLC_GEMM/$MODEL_NAME-q4f16_autoawq-cuda.so
killall nvidia-smi

#### Unstable
# MODEL_Q4F16_AWQ_MLC_GEMV=$METHOD_EXPORTS/q4f16_autoawq_gemv
# start_mem_tracker "quantize_q4f16_autoawq_gemv"
# mlc_chat convert_weight $MODEL_PATH \
#     --quantization q4f16_autoawq \
#     --source-format awq \
#     --source $AWQ_EXPORTS/gemv/model.safetensors \
#     -o $MODEL_Q4F16_AWQ_MLC_GEMV
# # 2. gen_config: generate mlc-chat-config.json and process tokenizers
# mlc_chat gen_config $MODEL_PATH \
#     --quantization q4f16_autoawq --conv-template "llama-2" \
#     -o $MODEL_Q4F16_AWQ_MLC_GEMV
# # 3. compile: compile model library with specification in mlc-chat-config.json
# mlc_chat compile $MODEL_Q4F16_AWQ_MLC_GEMV/mlc-chat-config.json \
#     --device cuda -o $MODEL_Q4F16_AWQ_MLC_GEMV/$MODEL_NAME-q4f16_autoawq-cuda.so
# killall nvidia-smi


# baseline
MODEL_Q0F16_1_MLC=$METHOD_EXPORTS/q0f16
start_mem_tracker "baseline"
mlc_chat convert_weight $MODEL_PATH \
    --quantization q0f16 \
    -o $MODEL_Q0F16_1_MLC
# 2. gen_config: generate mlc-chat-config.json and process tokenizers
mlc_chat gen_config $MODEL_PATH \
    --quantization q0f16 --conv-template "llama-2" \
    -o $MODEL_Q0F16_1_MLC
# 3. compile: compile model library with specification in mlc-chat-config.json
mlc_chat compile $MODEL_Q0F16_1_MLC/mlc-chat-config.json \
    --device cuda -o $MODEL_Q0F16_1_MLC/$MODEL_NAME-q0f16-cuda.so
killall nvidia-smi


MODEL_Q3F16_1_MLC=$METHOD_EXPORTS/q3f16_1
start_mem_tracker "quantize_q3f16_1"
mlc_chat convert_weight $MODEL_PATH \
    --quantization q3f16_1 \
    -o $MODEL_Q3F16_1_MLC
# 2. gen_config: generate mlc-chat-config.json and process tokenizers
mlc_chat gen_config $MODEL_PATH \
    --quantization q3f16_1 --conv-template "llama-2" \
    -o $MODEL_Q3F16_1_MLC
# 3. compile: compile model library with specification in mlc-chat-config.json
mlc_chat compile $MODEL_Q3F16_1_MLC/mlc-chat-config.json \
    --device cuda -o $MODEL_Q3F16_1_MLC/$MODEL_NAME-q3f16_1-cuda.so
killall nvidia-smi

# NOTE: To generate or bench omniquant models, one needs to incorporate the quantization schemes as mentioned towards the bottom in this notebook https://github.com/OpenGVLab/OmniQuant/blob/main/runing_quantized_models_with_mlc_llm.ipynb. 
# Will add a pr (or a fork link) with the changes which can then be built to support omniquant models.