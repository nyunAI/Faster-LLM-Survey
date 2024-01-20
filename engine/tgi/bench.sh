BASE=$(pwd)
METHOD=engine/tgi
METHOD_PATH=$METHOD/tgi

MODEL_NAME=Llama-2-7b-hf

EXPORTS=exports/$MODEL_NAME
METHOD_EXPORTS=$EXPORTS/$METHOD

AWQ_EXPORTS=$EXPORTS/quant/awq
GPTQ_EXPORTS=$EXPORTS/quant/gptq
SQUEEZELM_EXPORTS=model/squeezellm


echo "If using gated models: export HUGGINGFACE_HUB_TOKEN=<>"
# models
# awq
MODEL_AWQ_Q4_GEMM=$AWQ_EXPORTS/q4_gemm
MODEL_AWQ_Q4_GEMV=$AWQ_EXPORTS/q4_gemv
# gptq (with tokenizer.model and tokenizer.json copied from model)
MODEL_Q4_128G=$GPTQ_EXPORTS/q4_128g
MODEL_Q8_128G=$GPTQ_EXPORTS/q8_128g
# squeezelm

# git submodule update --init model/squeezellm/*
MODEL_SQUEEZELM_Q4=$SQUEEZELM_EXPORTS/w4-s0

cd $METHOD_PATH

for MODEL_TYPE in $MODEL_AWQ_Q4_GEMM $MODEL_AWQ_Q4_GEMV $MODEL_Q4_128G $MODEL_Q8_128G $MODEL_SQUEEZELM_Q4; do
    echo "Benchmarking $MODEL_TYPE"
done




# NOTE: tgi does not load autogptq output formats directly
# raise EnvironmentError(
# OSError: Error no file named pytorch_model.bin, tf_model.h5, model.ckpt.index or flax_model.msgpack found in directory /home/shwu/LLM-Efficiency-Survey/exports/Llama-2-7b-hf/quant/gptq/q4_128g. 