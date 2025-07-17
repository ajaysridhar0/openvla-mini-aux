sleep 10000
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
    -d \
    openvla-mini-aux \
    "bash_scripts/robocasa/mug_rep/panda_og+mg_pnp_chain.sh"
sleep 2750
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
    -d \
    openvla-mini-aux \
    "bash_scripts/robocasa/mug_rep/panda_og+mg_pnp_bbox.sh"
sleep 2750
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
    -d \
    openvla-mini-aux \
    "bash_scripts/robocasa/mug_rep/panda_og+mg_pnp_lm.sh"
sleep 2750
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
    -d \
    openvla-mini-aux \
    "bash_scripts/robocasa/mug_rep/panda_og+mg_pnp_ee.sh"
sleep 2750
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
    -d \
    openvla-mini-aux \
    "bash_scripts/robocasa/mug_rep/jaco+mg_pnp_chain.sh"
sleep 2750
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
    -d \
    openvla-mini-aux \
    "bash_scripts/robocasa/mug_rep/jaco+mg_pnp_bbox.sh"
sleep 2750
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
    -d \
    openvla-mini-aux \
    "bash_scripts/robocasa/mug_rep/jaco+mg_pnp_lm.sh"
sleep 2750
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
    -d \
    openvla-mini-aux \
    "bash_scripts/robocasa/mug_rep/jaco+mg_pnp_ee.sh"

