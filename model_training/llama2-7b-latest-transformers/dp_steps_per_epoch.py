import os
from datasets import load_from_disk, concatenate_datasets, Dataset

def compute_steps_per_epoch(dataset_path, mini_batch_size, cores, nodes, tensor_parallel_size):
    # Load the dataset
    dataset = load_from_disk(dataset_path)

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
dataset_path =  "~/examples_datasets/packed-hf-training-dataset-89B"
global_batch_size = 2048
mini_batch_size = 1 # Example mini-batch size
cores = 32  # Example number of cores
nodes = 16  # Example number of nodes
tensor_parallel_size = 8  # Example tensor parallel size
gradient_accumulation_steps = (global_batch_size / (mini_batch_size * (cores*nodes)/(tensor_parallel_size))) # Example gradient accumulation steps

steps_per_epoch = compute_steps_per_epoch(dataset_path, mini_batch_size, cores, nodes, tensor_parallel_size)/ gradient_accumulation_steps

print(f"Steps per epoch: {steps_per_epoch}")
