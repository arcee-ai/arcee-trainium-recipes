import os
from datasets import load_from_disk, concatenate_datasets, Dataset

def compute_steps_per_epoch(dataset_path, mini_batch_size, cores, nodes, tensor_parallel_size):
    # Load the dataset
    dataset = load_from_disk(dataset_path)

    # arrow_files = [file for file in os.listdir(dataset_path) if file.endswith(".arrow")]
    # dataset = concatenate_datasets([Dataset.from_file(dataset_path+arrow_file) for arrow_file in arrow_files])
    # print("dataset: ", dataset)

    # Count the number of training examples
    num_training_examples = len(dataset)

    # num_training_examples=37429629

    print(f"Number of training examples: {num_training_examples}")

    # Calculate data parallel size (DP)
    data_parallel_size = cores * nodes / tensor_parallel_size

    # Calculate examples per single pass
    examples_per_single_pass = data_parallel_size * mini_batch_size

    # Compute the number of steps per epoch
    steps_per_epoch = num_training_examples / examples_per_single_pass

    return steps_per_epoch

# Example usage
dataset_path =  "~/examples_datasets/packed-hf-training-dataset-89B"#'~/datasets/'
global_batch_size = 2048
mini_batch_size = 1 # Example mini-batch size
cores = 32  # Example number of cores
nodes = 16  # Example number of nodes
tensor_parallel_size = 8  # Example tensor parallel size
gradient_accumulation_steps = (global_batch_size / (mini_batch_size * (cores*nodes)/(tensor_parallel_size))) # Example gradient accumulation steps

steps_per_epoch = compute_steps_per_epoch(dataset_path, mini_batch_size, cores, nodes, tensor_parallel_size)/ gradient_accumulation_steps

print(f"Steps per epoch: {steps_per_epoch}")

#GBS=128
# Number of training examples: 37429629
# Steps per epoch: 292418.9765625

#GBS=256
# Number of training examples: 37429629
# Steps per epoch: 146209.48828125 

#GBS=1024
# Number of training examples: 37429629
# Steps per epoch: 36552.3720703125

#GBS=1024  mock dataset
# Number of training examples: 1588925
# Steps per epoch: 1551.6845703125

# GBS=2048
# Number of training examples: 21708072
# Steps per epoch: 10599.64453125



#scp -i   arcee-dev-trn-cluster.pem  ubuntu@ec2-54-80-251-40.compute-1.amazonaws.com:/fsx/updated_llama2_pretrain_loading/output/neuron_tblogs_021224_2331_bfloat16_w64_lr2e-05_bs1_acc16_warmup100_max5000_xlaTrue_trn1n.32xlarge/events.out.tfevents.1707780707.compute1-dy-queue1-i1-1.117695.0  ./my-logs-new
