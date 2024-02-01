# generates different outputs from llamacpp
METHOD=quant/omniquant
METHOD_PATH=$METHOD/omniquant

MODEL_NAME=Llama-2-7b-hf

MODEL_PATH=model/$MODEL_NAME

EXPORTS=exports/$MODEL_NAME
METHOD_EXPORTS=$EXPORTS/$METHOD
METHOD_EXPORTS_PPL=$EXPORTS/$METHOD/ppl

mkdir -p $METHOD_EXPORTS

# Function to start memory tracker
start_mem_tracker() {
    nvidia-smi --format=csv,nounits --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.free,memory.total -lms 100 > $METHOD_EXPORTS/gpu_bench_mem_usage_$1.csv &
}
# baseline

SCALE_PATH=$METHOD_EXPORTS/act_scales
SHIFT_PATH=$METHOD_EXPORTS/act_shifts

MODEL_W4A16=$METHOD_EXPORTS/w4a16
MODEL_W3A16=$METHOD_EXPORTS/w3a16

MODEL_W4A16G128=$METHOD_EXPORTS/w4a16g128
MODEL_W3A16G128=$METHOD_EXPORTS/w3a16g128


# bench with mlcllm/bench.sh
# remove c4 if eval on model. Has outdated data loader implementations


# following is for eval tasks
# start_mem_tracker "w4a16"
# python $METHOD_PATH/main.py \
# --model $MODEL_PATH \
# --act-scales $SCALE_PATH \
# --act-shifts $SHIFT_PATH \
# --calib_dataset wikitext2 \
# --epochs 0 --output_dir $MODEL_W4A16 \
# --eval_ppl --wbits 4 --abits 16 --lwc --tasks swag \
# --resume $MODEL_W4A16/omni_parameters.pth
# killall nvidia-smi


# ======================== PPL ========================
pretrained=$METHOD_EXPORTS/pretrained

CUDA_VISIBLE_DEVICES=0 python $METHOD_PATH/cal_ppl.py \
--model $MODEL_PATH  \
--epochs 0 --output_dir $METHOD_EXPORTS_PPL/w3a16g128 \
--eval_ppl --wbits 3 --abits 16 --group_size 128 --lwc \
--resume $pretrained/Llama-2-7b-w3a16g128.pth

CUDA_VISIBLE_DEVICES=0 python $METHOD_PATH/cal_ppl.py \
--model $MODEL_PATH  \
--epochs 0 --output_dir $METHOD_EXPORTS_PPL/w4a16g128 \
--eval_ppl --wbits 4 --abits 16 --group_size 128 --lwc \
--resume $pretrained/Llama-2-7b-w4a16g128.pth