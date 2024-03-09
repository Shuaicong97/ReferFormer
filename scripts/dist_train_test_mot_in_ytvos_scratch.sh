#!/usr/bin/env bash
set -x

GPUS=${GPUS:-8}
PORT=${PORT:-29500}
if [ $GPUS -lt 8 ]; then
    GPUS_PER_NODE=${GPUS_PER_NODE:-$GPUS}
else
    GPUS_PER_NODE=${GPUS_PER_NODE:-8}
fi
CPUS_PER_TASK=${CPUS_PER_TASK:-5}

OUTPUT_DIR=$1
PY_ARGS=${@:2}  # Any other arguments 

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)
            GPUS=$2
            shift 2
            ;;
        *)
            PY_ARGS+=("$1")
            shift
            ;;
    esac
done

# train
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${PORT} --use_env \
main.py --with_box_refine --binary --freeze_text_encoder --gpus=${GPUS} \
--output_dir=${OUTPUT_DIR} ${PY_ARGS}

# test
CHECKPOINT=${OUTPUT_DIR}/checkpoint.pth
python3 inference_mot_in_ytvos.py --with_box_refine --binary --freeze_text_encoder --gpus=${GPUS} \
--resume=${CHECKPOINT}  ${PY_ARGS}

