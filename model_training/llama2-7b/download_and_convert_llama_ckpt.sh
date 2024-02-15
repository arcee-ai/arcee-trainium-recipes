#!/bin/bash
# Save llama2-7b weights as a single pytorch bin file, which is required by checkpoint conversion script.
# Assumes that you have logged into Hugging Face hub using `huggingface-cli login`
cat <<EOF | python3
import transformers
model = transformers.AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model.save_pretrained("./llama2_7b_hf/", max_shard_size="50GB")
EOF

# Convert HF llama2-7b model weights to NxD sharded format
if [ -f ./llama2_7b_hf/pytorch_model.bin ]; then
  python3 convert_checkpoints.py \
  --input_dir ./llama2_7b_hf/pytorch_model.bin \
  --output_dir ./llama2_7b_hf_sharded \
  --convert_from_full_model \
  --tp_size 8
else
  echo "Error: HF llama2-7b checkpoint file not found"
fi
