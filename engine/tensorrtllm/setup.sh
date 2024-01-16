BASE=$(pwd)
METHOD=engine/tensorrtllm
METHOD_PATH=$METHOD/tensorrtllm

# we use the following docker image for tensorrtllm

docker pull shwunyunai/tensorrt-llm:triton_trt
# run the following to start the docker container. And then run the generate.sh script inside the container.

# docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it -v $BASE:/bench --workdir /bench shwunyunai/tensorrt-llm:triton_trt # bash $METHOD_PATH/generate.sh
# docker rm $(docker ps -aq)