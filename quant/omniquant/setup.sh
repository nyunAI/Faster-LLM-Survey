# for cuda12.1
METHOD=quant/omniquant
METHOD_PATH=$METHOD/omniquant

cd $METHOD_PATH
pip install -e .
# requires gptq kernel. install with quant/gptq/setup.sh