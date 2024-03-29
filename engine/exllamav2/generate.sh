# generates different outputs from exllamav2
METHOD=engine/exllamav2
METHOD_PATH=$METHOD/exllamav2

MODEL_NAME=Llama-2-7b-hf
MODEL_PATH=model/$MODEL_NAME
EXPORTS=exports/$MODEL_NAME
METHOD_EXPORTS=$EXPORTS/$METHOD

EXL2_TEMP=$METHOD_EXPORTS/temp

mkdir -p $METHOD_EXPORTS
mkdir -p $EXL2_TEMP

# Function to start memory tracker
start_mem_tracker() {
    nvidia-smi --format=csv,nounits --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.free,memory.total -lms 100 > $METHOD_EXPORTS/gpu_mem_usage_$1.csv &
}

# run measurement
start_mem_tracker "measurement"
python $METHOD_PATH/convert.py \
-i $MODEL_PATH \
-o $EXL2_TEMP \
-nr \
-om $METHOD_EXPORTS/measurement.json
killall nvidia-smi
rm -rf $EXL2_TEMP/*

# quantized 4.0 bpw
MODEL_4BPW_EXL2=$METHOD_EXPORTS/4bpw-exl2
start_mem_tracker "quantize_4bpw"
python $METHOD_PATH/convert.py \
    -i $MODEL_PATH \
    -o $EXL2_TEMP \
    -nr \
    -m $METHOD_EXPORTS/measurement.json \
    -cf $MODEL_4BPW_EXL2 \
    -b 4.0
killall nvidia-smi
rm -rf $EXL2_TEMP/*

# quantized 6bpw
MODEL_6BPW_EXL2=$METHOD_EXPORTS/6bpw-exl2
start_mem_tracker "quantize_6bpw"
python $METHOD_PATH/convert.py \
    -i $MODEL_PATH \
    -o $EXL2_TEMP \
    -nr \
    -m $METHOD_EXPORTS/measurement.json \
    -cf $MODEL_6BPW_EXL2 \
    -b 6.0
killall nvidia-smi
rm -rf $EXL2_TEMP