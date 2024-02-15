#!/bin/bash

#############################################
# User defined parameters and env vars

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

export NEURON_CC_FLAGS="--model-type transformer --distribution-strategy=llm-training --cache_dir=/fsx/neuron_compile_cache/"
export NEURON_FUSE_SOFTMAX=1

# Async Runtime
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=3

# HOST OOM
export MALLOC_ARENA_MAX=64

# TP degree
TP_DEGREE=8
# 0: bf16; 1: mixed precision
USE_MIX_PRECISION=0
# 0: use pure DP; 1: use ZeRO-1
USE_ZERO_1=1
# global batch size
GBS=2048
# micro batch size
MBS=1 # MSB>1 can help with speeding up the process, but is not yet supported. 
# number of steps to run
TOTAL_STEPS=31800 #for 3 epochs = 3*10600
# warmup steps
WARMUP_STEPS=100 #ratio=0.01*total steps for one epoch 0.01*10600 ~ 100
# learning rate
LR=1.5e-4 #1.5e-4 based on the CPT paper but based on LLama2 paper they are using 3e-4 for pretraining and 2e-5 for SFT. Here we are not pretraining from scratch so I would go for 2e-5
# model path
MODEL_PATH=$SCRIPT_DIR
# data path
DATA_PATH="~/examples_datasets/packed-hf-training-dataset-89B" # full packed PMC data
# sequence length
SEQ_LEN=4096 #for pmc packed tokenized 

#############################################

export NUM_NEURONCORES=32
NODE_ID=0
WORLD_SIZE=1
DISTRIBUTED_ARGS="--nproc_per_node $NUM_NEURONCORES"
if [ ! -z "$SLURM_NTASKS" ]; then
    WORLD_SIZE=$SLURM_NTASKS
    NODE_ID=$SLURM_NODEID
    MASTER_ADDRESS=(`scontrol show hostnames $SLURM_JOB_NODELIST`)
    DISTRIBUTED_ARGS="--nproc_per_node $NUM_NEURONCORES --nnodes $WORLD_SIZE --node_rank $NODE_ID --master_addr $MASTER_ADDRESS --master_port 44000"
    if [ $NODE_ID -eq 0 ]; then
        echo "WORLD_SIZE=$WORLD_SIZE"
        echo "NODE_ID=$NODE_ID"
        echo "MASTER_ADDRESS=$MASTER_ADDRESS"
        echo "DISTRIBUTED_ARGS=$DISTRIBUTED_ARGS"
    fi
    export FI_EFA_USE_DEVICE_RDMA=1
    export FI_PROVIDER=efa
fi

echo "WORLD_SIZE=$WORLD_SIZE"
echo "NODE_ID=$NODE_ID"
echo "MASTER_ADDRESS=$MASTER_ADDRESS"

sudo sysctl -w net.ipv4.ip_local_reserved_ports=44000,48620

export NEURON_RT_NUM_CORES=32
export NUM_NEURONCORES=$NEURON_RT_NUM_CORES
export TPU_NUM_DEVICES=$NEURON_RT_NUM_CORES
export TPU_CHIPS_PER_HOST_BOUNDS=$NEURON_RT_NUM_CORES
export NEURON_RT_ROOT_COMM_ID=localhost:48620

#############################################

EXTRA_ARGS=" "
if [ $USE_MIX_PRECISION -gt 0 ]; then
    EXTRA_ARGS+=" --use_mix_precision"
fi
if [ $USE_ZERO_1 -gt 0 ]; then
    EXTRA_ARGS+=" --use_zero_1"
fi

DP=$(($NEURON_RT_NUM_CORES * $WORLD_SIZE / $TP_DEGREE))
ACC_STEPS=$(($GBS / $MBS / $DP))


if [ $NEURON_EXTRACT_GRAPHS_ONLY -gt 0 ]; then
    STEPS_THIS_RUN=2
    OUTPUT_LOG=log_compile-$NODE_ID.log
else
    STEPS_THIS_RUN=-1
    OUTPUT_LOG=log_exe-$NODE_ID.log
fi

echo TP_DEGREE=$TP_DEGREE
echo USE_MIX_PRECISION=$USE_MIX_PRECISION
echo USE_ZERO_1=$USE_ZERO_1
echo GBS=$GBS
echo MBS=$MBS
echo TOTAL_STEPS=$TOTAL_STEPS
echo WARMUP_STEPS=$WARMUP_STEPS
echo LR=$LR
echo MODEL_PATH=$MODEL_PATH
echo DATA_PATH=$DATA_PATH
echo SEQ_LEN=$SEQ_LEN

echo EXTRA_ARGS=$EXTRA_ARGS
echo DP=$DP
echo ACC_STEPS=$ACC_STEPS
echo STEPS_THIS_RUN=$STEPS_THIS_RUN
echo OUTPUT_LOG=$OUTPUT_LOG

torchrun $DISTRIBUTED_ARGS \
    tp_zero1_llama2_7b_hf_pretrain.py \
    --model_path $MODEL_PATH \
    --data_dir $DATA_PATH \
    --tensor_parallel_size $TP_DEGREE \
    --batch_size $MBS \
    --steps_this_run $STEPS_THIS_RUN\
    --max_steps $TOTAL_STEPS \
    --warmup_steps $WARMUP_STEPS \
    --lr $LR \
    --grad_accum_usteps $ACC_STEPS \
    --seq_len $SEQ_LEN \
    --sequence_parallel_enabled \
    --selective_checkpoint_enabled \
    --logging_interval 10 \
    --checkpoint_dir "./checkpoints/llama2_7B_full_pmc" \
    --checkpoint_freq 10600 \
    --num_kept_checkpoint 3 \
    --load_pretrained_checkpoint \
    --pretrained_checkpoint_dir "llama2_7b_hf_sharded" \
    --separate_qkv \
    $EXTRA_ARGS |& tee $OUTPUT_LOG
