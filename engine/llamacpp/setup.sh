# requires make and nvidia-cuda-toolkit
METHOD=engine/llamacpp
METHOD_PATH=$METHOD/llama.cpp

cd $METHOD_PATH
# make LLAMA_CUBLAS=1
# pip install -r requirements.txt

cd awq-py
pip install -r requirements.txt