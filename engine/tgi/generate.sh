BASE=$(pwd)
METHOD=engine/tgi
METHOD_PATH=$METHOD/tgi

MODEL_NAME=Llama-2-7b-hf

EXPORTS=exports/$MODEL_NAME
METHOD_EXPORTS=$EXPORTS/$METHOD

AWQ_EXPORTS=$EXPORTS/quant/awq
GPTQ_EXPORTS=$EXPORTS/quant/gptq
SQUEEZELM_EXPORTS=model/squeezellm


text-generation-server quantize /home/shwu/LLM-Efficiency-Survey/model/Llama-2-7b-hf /home/shwu/LLM-Efficiency-Survey/exports/Llama-2-7b-hf/engine/tgi/gptq