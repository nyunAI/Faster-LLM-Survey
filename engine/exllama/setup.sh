# requires make and nvidia-cuda-toolkit
METHOD=engine/exllama
METHOD_PATH=$METHOD/exllama

cd $METHOD_PATH
pip install -r requirements.txt
