#!/bin/bash

# Convert NxD sharded and cpt llama2-7b model to full model
if [ -d ./~/checkpoints/step_4656/model/ ]; then
  python3 convert_checkpoints.py \
  --input_dir ./checkpoints/step_4656/model/ \
  --output_dir ./checkpoints/llama2-7B-3BTokens_4656/ \
  --convert_to_full_model \
  --tp_size 8 \
  --load_xser True

else
  echo "Error: Sharded cpt llama2-7b checkpoint directory not found."
fi
