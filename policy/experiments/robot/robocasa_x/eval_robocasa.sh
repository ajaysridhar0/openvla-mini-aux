export PYTHONPATH=/iliad/u/jenseng/xembod/openvla-mini-aux

ACT_HORIZON=8
ROLLOUT_DIR="experiments/rollouts/${MODEL}/${SAVE_FOLDER}"

MODEL="base--mg_pnp"
CHECKPOINT="step-150000-epoch-14-loss=0.7313.pt"
TASK="PnPCounterToSink"
AUX_TASK_TYPES=null
AUX_ABRV="act"
STEP="150k"

CHECKPOINT_PATH=runs/${MODEL}/checkpoints/${CHECKPOINT}
ACT_HORIZON=8

python experiments/robot/libero/run_robocasa_eval_new.py \
    --pretrained_checkpoint ${CHECKPOINT_PATH} \
    --robot PandaOmron \
    --gripper_types Robotiq85Gripper \
    --task ${TASK} \
    --unnorm_key mg_pnp \
    --act_horizon ${ACT_HORIZON} \
    --rollout_dir "experiments/rollouts/test/" \
    --aux_task_types ${AUX_TASK_TYPES}
