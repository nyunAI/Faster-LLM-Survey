## Load model

mkdir model && cd model
git lfs install
git clone https://huggingface.co/meta-llama/Llama-2-7b-hf


## Llama.cpp

bash llamacpp/setup.sh
bash llamacpp/generate.sh to generate ggml compiled files for fp16 and q4_0, q4_k_m quantized models
bash llamacpp/bench.sh to benchmark the above the 