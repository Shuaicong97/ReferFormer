#!/bin/bash -l

#SBATCH --job-name=train_with_ovis
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:8
#SBATCH --output=/home/atuin/v100dd/v100dd19/sbatch/result-%x-%j.txt
#SBATCH -C a100_80

set -x

OUTPUT_DIR=$1
GPUS=$2
PY_ARGS=${@:3}  # Any other arguments

GPUS=${GPUS:-8}
PORT=${PORT:-29500}
if [ $GPUS -lt 8 ]; then
    GPUS_PER_NODE=${GPUS_PER_NODE:-$GPUS}
else
    GPUS_PER_NODE=${GPUS_PER_NODE:-8}
fi
CPUS_PER_TASK=${CPUS_PER_TASK:-5}

# train
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${PORT} --use_env \
main.py --with_box_refine --binary --freeze_text_encoder \
--output_dir=${OUTPUT_DIR} --gpus=${GPUS} ${PY_ARGS}

# test
CHECKPOINT=${OUTPUT_DIR}/checkpoint.pth
python3 inference_ovis_in_ytvos.py --with_box_refine --binary --freeze_text_encoder --gpus=${GPUS} \
--resume=${CHECKPOINT} --output_dir=${OUTPUT_DIR} ${PY_ARGS}

