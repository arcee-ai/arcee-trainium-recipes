#!/bin/bash
sbatch --exclusive \
--nodes 1 \
--cpus-per-task 128 \
--wrap="srun bash $(pwd)/download_and_convert_llama_ckpt.sh"
