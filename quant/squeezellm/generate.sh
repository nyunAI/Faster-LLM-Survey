BASE=$(pwd)
METHOD=quant/squeezellm
METHOD_PATH=$METHOD/squeezellm
METHOD_SUBPATH=$METHOD/squeezellmgradients

MODEL_NAME=Llama-2-7b-hf
MODEL_PATH=model/$MODEL_NAME
EXPORTS=exports/$MODEL_NAME
METHOD_EXPORTS=$EXPORTS/$METHOD

DATASET_PATH=datasets/$DATASET_NAME/c4

mkdir -p $METHOD_EXPORTS

# Function to start memory tracker
start_mem_tracker() {
    nvidia-smi --format=csv,nounits --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.free,memory.total -lms 100 > $METHOD_EXPORTS/gpu_mem_usage_$1.csv &
}

GRADIENT_PATH=$METHOD_EXPORTS/gradient
# generate gradients
# start_mem_tracker "gradient" ## not needed as data loading might be very very high
CUDA_VISIBLE_DEVICES=-1 python $METHOD_SUBPATH/run.py --output_dir $GRADIENT_PATH --model_name_or_path $MODEL_PATH --dataset wikitext2
# killall nvidia-smi



