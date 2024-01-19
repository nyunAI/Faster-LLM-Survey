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
    nvidia-smi --format=csv,nounits --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.free,memory.total -lms 100 > $METHOD_EXPORTS/gpu_mem_usage_$1.csv &
}


# base engine
BASE_F16=$METHOD_EXPORTS/base_f16
mkdir -p $BASE_F16
start_mem_tracker "convert_base_f16"
python $METHOD_SUBPATH/build.py \
--model_dir $MODEL_PATH \
--dtype float16 \
--remove_input_padding \
--use_gpt_attention_plugin float16 \
--enable_context_fmha \
--use_gemm_plugin float16 \
--output_dir $BASE_F16/engine
killall nvidia-smi

# int8
MODEL_INT8=$METHOD_EXPORTS/int8
mkdir -p $MODEL_INT8
start_mem_tracker "convert_int8"
python $METHOD_SUBPATH/build.py \
--model_dir $MODEL_PATH \
--dtype float16 \
--remove_input_padding \
--use_gpt_attention_plugin float16 \
--enable_context_fmha \
--use_gemm_plugin float16 \
--use_weight_only \
--output_dir $MODEL_INT8/engine
killall nvidia-smi

# awq
MODEL_INT4_AWQ=$METHOD_EXPORTS/int4_awq
mkdir -p $MODEL_INT4_AWQ
start_mem_tracker "convert_int4_awq"
python $METHOD_SUBPATH/quantize.py \
--model_dir $MODEL_PATH \
--dtype float16 \
--qformat int4_awq \
--export_path $MODEL_INT4_AWQ/llama-7b-4bit-gs128-awq.pt \
--calib_size 32

python $METHOD_SUBPATH/build.py \
--model_dir $MODEL_PATH \
--quant_ckpt_path $MODEL_INT4_AWQ/llama-7b-4bit-gs128-awq.pt \
--dtype float16 \
--remove_input_padding \
--use_gpt_attention_plugin float16 \
--enable_context_fmha \
--use_gemm_plugin float16 \
--use_weight_only \
--weight_only_precision int4_awq \
--per_group \
--output_dir $MODEL_INT4_AWQ/engine
killall nvidia-smi


# smoothquant
MODEL_SQ0_8=$METHOD_EXPORTS/sq0_8
mkdir -p $MODEL_SQ0_8
start_mem_tracker "convert_sq0_8"
python $METHOD_SUBPATH/hf_llama_convert.py \
-i $MODEL_PATH \
-o $MODEL_SQ0_8 \
-sq 0.8 --storage-type fp16

python $METHOD_SUBPATH/build.py \
--ft_model_dir $MODEL_SQ0_8/1-gpu/ \
--use_smooth_quant \
--output_dir $MODEL_SQ0_8/engine
killall nvidia-smi

# gptq
GPTQ_EXPORTS=$EXPORTS/quant/gptq
MODEL_GPTQ=$METHOD_EXPORTS/gptq
mkdir -p $MODEL_GPTQ
start_mem_tracker "convert_gptq"
python $METHOD_SUBPATH/build.py \
--model_dir $MODEL_PATH \
--quant_ckpt_path $GPTQ_EXPORTS/q4_128g/gptq_model-4bit-128g.safetensors \
--dtype float16 \
--remove_input_padding \
--use_gpt_attention_plugin float16 \
--enable_context_fmha \
--use_gemm_plugin float16 \
--use_weight_only \
--weight_only_precision int4_gptq \
--per_group \
--output_dir $MODEL_GPTQ/engine
killall nvidia-smi
