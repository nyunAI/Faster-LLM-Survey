BASE=$(pwd)
METHOD=engine/mlcllm
METHOD_PATH=$METHOD/mlcllm


cd $METHOD_PATH

# install tvm https://llm.mlc.ai/docs/install/tvm.html#id5
pip install --pre -U -f https://mlc.ai/wheels mlc-ai-nightly-cu121

# install mlc-llm https://llm.mlc.ai/docs/install/mlc_llm.html#id4
pip install --pre -U -f https://mlc.ai/wheels mlc-chat-nightly-cu121 mlc-ai-nightly-cu121