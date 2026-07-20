export PYTHONPATH=/iliad/u/jenseng/xembod/openvla-mini-aux

MODEL=$1
CHECKPOINT=$2
AUX_TASK_TYPES=$3
SAVE_FOLDER=$4
RUN_ID=$5

CHECKPOINT_PATH=runs/${MODEL}/checkpoints/${CHECKPOINT}

ACT_HORIZON=8
ROLLOUT_DIR="rollouts/${MODEL}/${SAVE_FOLDER}"

python experiments/robot/libero/run_robocasa_eval_new.py \
    --pretrained_checkpoint ${CHECKPOINT_PATH} \
    --robot PandaOmron \
    --task PnPCounterToSink \
    --unnorm_key panda_pnp_counter_to_sink \
    --obj_groups carrot \
    --act_horizon ${ACT_HORIZON} \
    --rollout_dir ${ROLLOUT_DIR} \
    --obj_xinit_range 0.15 \
    --obj_yinit_range 0.2 \
    --aux_task_types ${AUX_TASK_TYPES} \
    --run_id ${RUN_ID} \
    --generative_textures True \
    # --use_wrist_image True \