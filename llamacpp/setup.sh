# requires make and nvidia-cuda-toolkit
METHOD=llamacpp
METHOD_PATH=$METHOD/llama.cpp

git clone https://github.com/ggerganov/llama.cpp $METHOD_PATH
cd $METHOD_PATH
make LLAMA_CUBLAS=1
pip install -r requirements.txt