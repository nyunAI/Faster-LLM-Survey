# generates different outputs from llmpruner
METHOD=prune/llmpruner
METHOD_PATH=$METHOD/llmpruner

MODEL_NAME=Llama-2-7b-hf
MODEL_PATH=model/$MODEL_NAME
EXPORTS=exports/$MODEL_NAME
METHOD_EXPORTS=$EXPORTS/$METHOD

DATASET_NAME=alpaca-cleaned
DATASET_PATH=datasets/$DATASET_NAME

mkdir -p $METHOD_EXPORTS

# Function to start memory tracker
start_mem_tracker() {
    nvidia-smi --format=csv,nounits --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.free,memory.total -lms 100 > $METHOD_EXPORTS/gpu_mem_usage_$1.csv &
}


# change hf_prune.py#32 to the following
# ```python
# logger = LoggerWithDepth(
#     env_name="", 
#     config=args.__dict__,
#     root_dir=args.save_ckpt_log_name,
#     setup_sublogger=True
# )
# ```

# quantize awq-gemm
MODEL_PRUNED_0_20=$METHOD_EXPORTS/prune_0_20
start_mem_tracker "prune_0_20"
python $METHOD_PATH/hf_prune.py \
--base_model $MODEL_PATH \
--pruning_ratio 0.20 \
--device cpu  \
--eval_device cuda \
--block_wise \
--block_mlp_layer_start 4 \
--block_mlp_layer_end 30 \
--block_attention_layer_start 4 \
--block_attention_layer_end 30 \
--save_ckpt_log_name $MODEL_PRUNED_0_20 \
--pruner_type taylor \
--test_after_train \
--taylor param_first \
--save_model
killall nvidia-smi


# use the docker image(tensorrtllm_trt:transformer_patched) instead

# # post-training
# MODEL_TUNE_0_20=$MODEL_PRUNED_0_20
# start_mem_tracker "tune_0_20"
# python $METHOD_PATH/post_training.py \
# --base_model $MODEL_PATH \
# --prune_model $MODEL_TUNE_0_20/pytorch_model.bin \
# --data_path $DATASET_PATH \
# --output_dir $MODEL_TUNE_0_20 \
# --wandb_project llama_tune \
# --lora_r 8 \
# --num_epochs 2 \
# --learning_rate 1e-4 \
# --batch_size 64
# killall nvidia-smi