METHOD=engine/vllm
METHOD_PATH=$METHOD

MODEL_NAME=Llama-2-7b-hf
DATASET_NAME=alpaca-cleaned

MODEL_PATH=model/$MODEL_NAME
DATASET_PATH=datasets/$DATASET_NAME/alpaca_data_cleaned.json

EXPORTS=exports/$MODEL_NAME
METHOD_EXPORTS=$EXPORTS/$METHOD

mkdir -p $METHOD_EXPORTS

AWQ_EXPORTS=$EXPORTS/quant/awq
GPTQ_EXPORTS=$EXPORTS/quant/gptq
SQUEEZELM_EXPORTS=model/squeezellm


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
# MODEL_SQUEEZELM_Q8=$SQUEEZELM_EXPORTS/sq-llama-2-7b-w3-s0.pt


echo "generate the quantized outputs from gptq, awq, squeezellm"