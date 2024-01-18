METHOD=engine/exllamav2
METHOD_PATH=$METHOD/exllamav2

MODEL_NAME=Llama-2-7b-hf
MODEL_PATH=model/$MODEL_NAME
EXPORTS=exports/$MODEL_NAME
METHOD_EXPORTS=$EXPORTS/$METHOD

GPTQ_EXPORTS=$EXPORTS/quant/gptq

mkdir -p $METHOD_EXPORTS

# Function to start memory tracker
start_mem_tracker() {
    nvidia-smi --format=csv,nounits --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.free,memory.total -lms 100 > $METHOD_EXPORTS/gpu_bench_mem_usage_$1.csv &
}

sleep 3
start_mem_tracker "baseline"
sleep 3
python $METHOD_PATH/test_inference.py -gs 40 -m $MODEL_PATH -s 2>&1 | tee $METHOD_EXPORTS/bench_baseline.log
killall nvidia-smi

# List of model types
MODEL_4BPW_EXL2="4bpw-exl2"
# MODEL_6BPW_EXL2="6bpw-exl2"
MODEL_8BPW_EXL2="8bpw-exl2"


# Loop through each model type
for MODEL_TYPE in "$MODEL_4BPW_EXL2" "$MODEL_8BPW_EXL2"; do
    MODEL_PATH=$METHOD_EXPORTS/$MODEL_TYPE

    sleep 3
    start_mem_tracker $MODEL_TYPE
    sleep 3
    # Run llama-bench for the current model type
    python $METHOD_PATH/test_inference.py -gs 40 -m $MODEL_PATH -s 2>&1 | tee $METHOD_EXPORTS/bench_$MODEL_TYPE.log

    # Stop memory tracker for the current model type
    killall nvidia-smi
done

# Loop through each model type
for MODEL_TYPE in "$MODEL_4BPW_EXL2" "$MODEL_8BPW_EXL2"; do
    MODEL_PATH=$METHOD_EXPORTS/$MODEL_TYPE

    sleep 3
    start_mem_tracker "$MODEL_TYPE"_prompt

    # Run llama-bench for the current model type
    python $METHOD_PATH/test_inference.py -gs 40 -m $MODEL_PATH -p "What is the meaning of life?" 2>&1 | tee $METHOD_EXPORTS/bench_prompt_$MODEL_TYPE.log

    # Stop memory tracker for the current model type
    killall nvidia-smi
done

# autogptq
MODEL_Q4_128G=q4_128g
MODEL_Q8_128G=q8_128g # not supported

cp $MODEL_PATH/tokenizer.model $GPTQ_EXPORTS/$MODEL_Q4_128G/tokenizer.model
cp $MODEL_PATH/tokenizer.model $GPTQ_EXPORTS/$MODEL_Q8_128G/tokenizer.model

cp $MODEL_PATH/tokenizer.json $GPTQ_EXPORTS/$MODEL_Q4_128G/tokenizer.json
cp $MODEL_PATH/tokenizer.json $GPTQ_EXPORTS/$MODEL_Q8_128G/tokenizer.json

# Loop through each model type
for MODEL_TYPE in "$MODEL_Q4_128G"; do
    MODEL_PATH=$GPTQ_EXPORTS/$MODEL_TYPE

    sleep 3
    start_mem_tracker "$MODEL_TYPE"_gptq

    # Run llama-bench for the current model type
    python $METHOD_PATH/test_inference.py -gs 40 -m $MODEL_PATH -s 2>&1 | tee $METHOD_EXPORTS/bench_$MODEL_TYPE.log

    # Stop memory tracker for the current model type
    killall nvidia-smi
done

for MODEL_TYPE in "$MODEL_Q4_128G"; do
    MODEL_PATH=$GPTQ_EXPORTS/$MODEL_TYPE

    sleep 3
    start_mem_tracker "$MODEL_TYPE"_gptq_prompt

    # Run llama-bench for the current model type
    python $METHOD_PATH/test_inference.py -gs 40 -m $MODEL_PATH -p "What is the meaning of life?" 2>&1 | tee $METHOD_EXPORTS/bench_prompt_$MODEL_TYPE.log

    # Stop memory tracker for the current model type
    killall nvidia-smi
done