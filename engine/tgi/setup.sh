BASE=$(pwd)
METHOD=engine/tgi
METHOD_PATH=$METHOD/tgi


# # installs protoc
# PROTOC_ZIP=protoc-21.12-linux-x86_64.zip
# curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v21.12/$PROTOC_ZIP
# sudo unzip -o $PROTOC_ZIP -d /usr/local bin/protoc
# sudo unzip -o $PROTOC_ZIP -d /usr/local 'include/*'
# rm -f $PROTOC_ZIP

# # installs rust
# curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# # installs openssl
# sudo apt-get install libssl-dev gcc -y

cd $METHOD_PATH
BUILD_EXTENSIONS=True make install make install

pip install autoawq
pip install optimum auto-gptq