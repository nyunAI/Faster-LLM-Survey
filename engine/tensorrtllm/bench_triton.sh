# runs inside the tensorrtllm docker container

BASE=$(pwd)
METHOD=engine/tensorrtllm
METHOD_PATH=$METHOD/tensorrtllm
METHOD_SUBPATH=/app/tensorrt_llm/examples/llama

TENSORRTLLM_BACKEND=$METHOD/tensorrtllm_backend

MODEL_NAME=Llama-2-7b-hf
MODEL_PATH=model/$MODEL_NAME
EXPORTS=exports/$MODEL_NAME
METHOD_EXPORTS=$EXPORTS/$METHOD

DATASET_NAME=alpaca-cleaned
DATASET_PATH=datasets/$DATASET_NAME

mkdir -p $METHOD_EXPORTS

# apt-get update
# apt-get install psmisc

# Function to start memory tracker
start_mem_tracker() {
    nvidia-smi --format=csv,nounits --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.free,memory.total -lms 100 > $METHOD_EXPORTS/gpu_bench_mem_usage_$1.csv &
}

# start_mem_tracker "baseline"
# sleep 3
# python $METHOD_PATH/benchmarks/python/benchmark.py \
# -m llama_7b \
# --mode plugin \
# --batch_size "1" \
# --input_output_len "128,128" 2>&1 | tee $METHOD_EXPORTS/benchmark_baseline.log
# killall nvidia-smi
# sleep 3


BASE_F16=base_f16
MODEL_INT8=int8
MODEL_INT4_AWQ=int4_awq # error
MODEL_SQ0_8=sq0_8
MODEL_GPTQ=gptq

git submodule deinit -f $TENSORRTLLM_BACKEND
git submodule update --init -f $TENSORRTLLM_BACKEND

# for MODEL_TYPE in "$MODEL_INT4_AWQ" "$BASE_F16" "$MODEL_INT8" "$MODEL_SQ0_8" "$MODEL_GPTQ" ; do
for MODEL_TYPE in "$MODEL_INT4_AWQ"; do
    MODEL_ENGINE_PATH=$METHOD_EXPORTS/$MODEL_TYPE/engine

    # copy the engine to the backend
    d=$TENSORRTLLM_BACKEND/all_models/$MODEL_TYPE
    mkdir -p $d
    cp -r $TENSORRTLLM_BACKEND/all_models/inflight_batcher_llm/* $d
    echo "Copying engine files for $MODEL_TYPE to backend"
    cp $MODEL_ENGINE_PATH/* $TENSORRTLLM_BACKEND/all_models/$MODEL_TYPE/tensorrt_llm/1/
    echo "Copying engine files for $MODEL_TYPE to backend done"
    # change the config file
    python $TENSORRTLLM_BACKEND/tools/fill_template.py --in_place "$TENSORRTLLM_BACKEND/all_models/$MODEL_TYPE/tensorrt_llm/config.pbtxt" "decoupled_mode:true,engine_dir:/all_models/$MODEL_TYPE/tensorrt_llm/1,max_tokens_in_paged_kv_cache:,batch_scheduler_policy:guaranteed_completion,kv_cache_free_gpu_mem_fraction:0.2,max_num_sequences:4"

    # modify config for the preprocessing component
    python $TENSORRTLLM_BACKEND/tools/fill_template.py --in_place "$TENSORRTLLM_BACKEND/all_models/$MODEL_TYPE/preprocessing/config.pbtxt" "tokenizer_type:llama,tokenizer_dir:meta-llama/Llama-2-7b-hf"

    # modify config for the postprocessing component
    python $TENSORRTLLM_BACKEND/tools/fill_template.py --in_place "$TENSORRTLLM_BACKEND/all_models/$MODEL_TYPE/postprocessing/config.pbtxt" "tokenizer_type:llama,tokenizer_dir:meta-llama/Llama-2-7b-hf"
done

echo -e "\
****************\\n\
****************\\n\
Run the following command inside docker:\\n\
pip install sentencepiece protobuf\\n\
CUDA_VISIBLE_DEVICES=0 python /opt/scripts/launch_multi_triton_servers.py --world_size 1 --model_repos /all_models/*\\n\
****************\\n\
****************"

# cd $TENSORRTLLM_BACKEND

sudo docker run -it --rm --network host --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --gpus all \
-v $BASE/$TENSORRTLLM_BACKEND/all_models:/all_models \
-v $BASE/$METHOD/scripts:/opt/scripts \
-v $BASE/$MODEL_PATH:/$MODEL_PATH \
nvcr.io/nvidia/tritonserver:23.10-trtllm-python-py3 bash

# run the following in the docker container

