# for cuda12.1
BASE=$(pwd)
METHOD=quant/omniquant
METHOD_PATH=$METHOD/omniquant
GPTQ_METHOD_PATH=$METHOD/gptq

cd $METHOD_PATH
pip install --upgrade pip
pip install -e .
cd $BASE


cd $GPTQ_METHOD_PATH
pip install -v .

# requires gptq kernel. install with quant/gptq/setup.sh