#!/bin/bash
sbatch --exclusive \
--nodes 1 \
--cpus-per-task 128 \
--wrap="srun bash $(pwd)/convert_to_full_model.sh"
