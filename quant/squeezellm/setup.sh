# requires make and nvidia-cuda-toolkit
BASE=$(pwd)
METHOD=quant/squeezellm
METHOD_PATH=$METHOD/squeezellm
METHOD_SUBPATH=$METHOD/squeezellmgradients

# install squeezellm
cd $METHOD_PATH
pip install -e .
cd squeezellm
python setup_cuda.py install
pip install scikit-learn==1.3.1
cd $BASE

# install squeezellmgradients
cd $METHOD_SUBPATH
pip install -e .
pip install -r requirements.txt