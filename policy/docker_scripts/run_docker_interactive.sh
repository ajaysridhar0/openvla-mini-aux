docker run --privileged --gpus all \
    --shm-size=1g \
    --ipc=host \
    --network=host \
    --env NCCL_DEBUG=INFO \
    --env NCCL_SOCKET_IFNAME=^docker0,lo-it \
    --rm \
    -it \
    -v /home/jensen.gao:/workspace \
    -v /datasets:/datasets \
    -w /workspace/openvla-mini-aux/ \
    openvla-mini-aux \
