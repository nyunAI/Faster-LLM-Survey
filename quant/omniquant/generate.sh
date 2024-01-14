METHOD=quant/omniquant
METHOD_PATH=$METHOD/omniquant

MODEL_NAME=Llama-2-7b-hf
DATASET_NAME=alpaca-cleaned

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

# generate activation scales and shifts
SCALE_PATH=$METHOD_EXPORTS/act_scales
SHIFT_PATH=$METHOD_EXPORTS/act_shifts

start_mem_tracker "activation_scales_shifts"
python $METHOD_PATH/generate_act_scale_shift.py \
--model $MODEL_PATH  \
--scales-output-path $SCALE_PATH \
--shifts-output-path $SHIFT_PATH \
--calib_dataset wikitext2
killall nvidia-smi

# generate quantized model
# W4A16
MODEL_W4A16=$METHOD_EXPORTS/w4a16
start_mem_tracker "quantize_w4a16"
python $METHOD_PATH/main.py \
--model $MODEL_PATH \
--epochs 1 \
--output_dir $MODEL_W4A16 \
--calib_dataset wikitext2 \
--act-scales $SCALE_PATH \
--act-shifts $SHIFT_PATH \
--wbits 4 --abits 16 --lwc
killall nvidia-smi

# W3A16
MODEL_W3A16=$METHOD_EXPORTS/w3a16
start_mem_tracker "quantize_w3a16"
python $METHOD_PATH/main.py \
--model $MODEL_PATH \
--epochs 1 \
--output_dir $MODEL_W3A16 \
--calib_dataset wikitext2 \
--act-scales $SCALE_PATH \
--act-shifts $SHIFT_PATH \
--wbits 3 --abits 16 --lwc
killall nvidia-smi