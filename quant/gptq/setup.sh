# for cuda12.1
METHOD=quant/gptq
METHOD_PATH=$METHOD

pip install auto-gptq
# for other cuda versions, see https://github.com/PanQiWei/AutoGPTQ/blob/main/README.md#quick-installation

# TODO: setup triton to quantize with triton
