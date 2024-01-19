METHOD=engine/vllm
METHOD_PATH=$METHOD

MODEL_NAME=Llama-2-7b-hf
DATASET_NAME=alpaca-cleaned

MODEL_PATH=model/$MODEL_NAME
DATASET_PATH=datasets/$DATASET_NAME/alpaca_data_cleaned.json

EXPORTS=exports/$MODEL_NAME
METHOD_EXPORTS=$EXPORTS/$METHOD

AWQ_EXPORTS=$EXPORTS/quant/awq
GPTQ_EXPORTS=$EXPORTS/quant/gptq

# git submodule update --init model/squeezellm
SQUEEZELM_EXPORTS=model/squeezellm

mkdir -p $METHOD_EXPORTS

# Function to start memory tracker
start_mem_tracker() {
    nvidia-smi --format=csv,nounits --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.free,memory.total -lms 100 > $METHOD_EXPORTS/gpu_bench_mem_usage_$1.csv &
}

# models
# awq
MODEL_AWQ_Q4_GEMM=$AWQ_EXPORTS/q4_gemm
MODEL_AWQ_Q4_GEMV=$AWQ_EXPORTS/q4_gemv # not yet supported ootb
# gptq (with tokenizer.model and tokenizer.json copied from model)
MODEL_Q4_128G=$GPTQ_EXPORTS/q4_128g
MODEL_Q8_128G=$GPTQ_EXPORTS/q8_128g # not yet supported ootb
# squeezelm
MODEL_SQUEEZELM_Q4=$SQUEEZELM_EXPORTS/w4-s0
MODEL_SQUEEZELM_Q3=$SQUEEZELM_EXPORTS/w3-s0 # not supported

# unsqueeze squeezellm models



# baseline
start_mem_tracker "baseline"
python $METHOD_PATH/benchmark_throughput.py --backend vllm --dataset $DATASET_PATH --model $MODEL_PATH --tokenizer $MODEL_PATH --num-prompts=1 2>&1 | tee $METHOD_EXPORTS/bench_baseline.log
killall nvidia-smi
sleep 3

# Loop through each model type
for QUANT_MODEL_PATH in "$MODEL_AWQ_Q4_GEMM" "$MODEL_Q4_128G"; do

    QUANT_TYPE=$(echo $QUANT_MODEL_PATH | cut -d'/' -f4)
    MODEL_STR="$QUANT_TYPE"_$(basename "$QUANT_MODEL_PATH" | sed 's/^.*\///')

    sleep 3
    start_mem_tracker $MODEL_STR
    # Run llama-bench for the current model type
    python $METHOD_PATH/benchmark_throughput.py --backend vllm --dataset $DATASET_PATH --quantization $QUANT_TYPE --model $QUANT_MODEL_PATH --tokenizer $QUANT_MODEL_PATH --num-prompts=1 2>&1 | tee $METHOD_EXPORTS/bench_$MODEL_STR.log

    # Stop memory tracker for the current model type
    killall nvidia-smi
    sleep 3
done

# squeezellm
# w4 s0
start_mem_tracker "squeezellm_w4_s0"
sleep 3
python $METHOD_PATH/benchmark_throughput.py --backend vllm --dataset $DATASET_PATH --quantization squeezellm --model $MODEL_SQUEEZELM_Q4 --tokenizer $MODEL_SQUEEZELM_Q4 --num-prompts=1 2>&1 | tee $METHOD_EXPORTS/bench_squeezellm_w4_s0.log
killall nvidia-smi
sleep 3