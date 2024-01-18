# built with cuda12.1, python3.10
METHOD=engine/exllamav2
METHOD_PATH=$METHOD/exllamav2

cd $METHOD_PATH
pip install -r requirements.txt
python setup.py install
# pip install exllamav2-0.0.4+cu118-cp310-cp310-linux_x86_64.whl

