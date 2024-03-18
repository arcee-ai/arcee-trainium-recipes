#!/bin/bash
sbatch --exclusive \
--nodes 16 \
--cpus-per-task 128 \
--wrap="srun neuron_parallel_compile bash $(pwd)/tp_zero1_llama2_7b_hf_pretrain.sh"
