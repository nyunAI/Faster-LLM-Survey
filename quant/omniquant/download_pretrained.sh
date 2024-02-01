METHOD=quant/omniquant
METHOD_PATH=$METHOD/omniquant

MODEL_NAME=Llama-2-7b-hf
DATASET_NAME=alpaca-cleaned

MODEL_PATH=model/$MODEL_NAME

EXPORTS=exports/$MODEL_NAME
METHOD_EXPORTS=$EXPORTS/$METHOD

mkdir -p $METHOD_EXPORTS

pretrained=$METHOD_EXPORTS/pretrained
# cd $pretrained
# wget "https://huggingface.co/ChenMnZ/OmniQuant/resolve/main/Llama-2-7b-w4a16g128.pth"
# wget "https://huggingface.co/ChenMnZ/OmniQuant/resolve/main/Llama-2-7b-w3a16g128.pth"

# download ppl script and make necessary changes
wget https://raw.githubusercontent.com/OpenGVLab/OmniQuant/main/main.py -O $METHOD_PATH/cal_ppl.py