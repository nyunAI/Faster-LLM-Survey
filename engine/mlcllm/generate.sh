METHOD=engine/mlcllm
METHOD_PATH=$METHOD/mlcllm

MODEL_NAME=Llama-2-7b-hf

MODEL_PATH=model/$MODEL_NAME

EXPORTS=exports/$MODEL_NAME
METHOD_EXPORTS=$EXPORTS/$METHOD

OMNIQUANT_TEMP=$METHOD_EXPORTS/temp

mkdir -p $METHOD_EXPORTS
mkdir -p $OMNIQUANT_TEMP

# Function to start memory tracker
start_mem_tracker() {
    nvidia-smi --format=csv,nounits --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.free,memory.total -lms 100 > $METHOD_EXPORTS/gpu_mem_usage_$1.csv &
}



# generate quantized model
# 1. Convert weight q4f16_1
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

# # 1. Convert weight q4f16_awq
# MODEL_Q4F16_AWQ_MLC=$METHOD_EXPORTS/q4f16_autoawq
# start_mem_tracker "quantize_q4f16_autoawq"
# mlc_chat convert_weight $MODEL_PATH \
#     --quantization q4f16_autoawq \
#     -o $MODEL_Q4F16_AWQ_MLC
# # 2. gen_config: generate mlc-chat-config.json and process tokenizers
# mlc_chat gen_config $MODEL_PATH \
#     --quantization q4f16_autoawq --conv-template "llama-2" \
#     -o $MODEL_Q4F16_AWQ_MLC
# # 3. compile: compile model library with specification in mlc-chat-config.json
# mlc_chat compile $MODEL_Q4F16_AWQ_MLC/mlc-chat-config.json \
#     --device cuda -o $MODEL_Q4F16_AWQ_MLC/$MODEL_NAME-q4f16_autoawq-cuda.so
# killall nvidia-smi

# # baseline
# MODEL_Q0F16_1_MLC=$METHOD_EXPORTS/q0f16
# start_mem_tracker "baseline"
# mlc_chat convert_weight $MODEL_PATH \
#     --quantization q0f16 \
#     -o $MODEL_Q0F16_1_MLC
# # 2. gen_config: generate mlc-chat-config.json and process tokenizers
# mlc_chat gen_config $MODEL_PATH \
#     --quantization q0f16 --conv-template "llama-2" \
#     -o $MODEL_Q0F16_1_MLC
# # 3. compile: compile model library with specification in mlc-chat-config.json
# mlc_chat compile $MODEL_Q0F16_1_MLC/mlc-chat-config.json \
#     --device cuda -o $MODEL_Q0F16_1_MLC/$MODEL_NAME-q0f16-cuda.so
# killall nvidia-smi

# NOTE: To generate or bench omniquant models, one needs to incorporate the quantization schemes as mentioned towards the bottom in this notebook https://github.com/OpenGVLab/OmniQuant/blob/main/runing_quantized_models_with_mlc_llm.ipynb. 
# Will add a pr (or a fork link) with the changes which can then be built to support omniquant models.