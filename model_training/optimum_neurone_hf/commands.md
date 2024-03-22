# Env Variables

NUM_WORKERS=32
TP_SIZE=8
MODEL_NAME=arcee-ai/Mistral-7B-Instruct-v0.2-expanded
TRAIN_BS=1
GRAD_ACCUM=1
BLOCK_SIZE=4096

 #mistralai/Mistral-7B-Instruct-v0.2

# Pre-compilation
WANDB_MODE=disabled neuron_parallel_compile torchrun --nproc_per_node=$NUM_WORKERS examples/language-modeling/run_clm.py         --model_name_or_path $MODEL_NAME         --dataset_name wikitext         --dataset_config_name wikitext-2-raw-v1         --do_train          --per_device_train_batch_size $TRAIN_BS         --gradient_accumulation_steps $GRAD_ACCUM         --block_size $BLOCK_SIZE         --tensor_parallel_size $TP_SIZE         --bf16          --zero_1         --logging_steps 1         --save_steps -1         --output_dir ~/test_extended_mistral_7b  


# Trainign

'''
WANDB_NAME=mistal-7b torchrun --nproc_per_node=$NUM_WORKERS examples/language-modeling/run_clm.py         --model_name_or_path $MODEL_NAME         --dataset_name wikitext         --dataset_config_name wikitext-2-raw-v1         --do_train          --per_device_train_batch_size $TRAIN_BS         --gradient_accumulation_steps $GRAD_ACCUM         --block_size $BLOCK_SIZE         --tensor_parallel_size $TP_SIZE         --bf16          --zero_1         --logging_steps 1         --save_steps -1         --output_dir ~/test_extended_mistral_7b          --overwrite_output_dir
'''