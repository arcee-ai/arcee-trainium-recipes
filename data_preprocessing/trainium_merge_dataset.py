from datasets import concatenate_datasets, load_from_disk
import os

# Path to the main folder containing all sub-folders with the datasets
main_folder_path = '/home/ubuntu/medical_packed_4096'

# List all sub-folders
sub_folders = [os.path.join(main_folder_path, f) for f in os.listdir(main_folder_path) if os.path.isdir(os.path.join(main_folder_path, f))]

# Load datasets from each sub-folder and collect them in a list
datasets_list = [load_from_disk(folder) for folder in sub_folders]

# Concatenate all datasets into one
full_dataset = concatenate_datasets(datasets_list)

# Now you can work with `full_dataset` as a single dataset
print(full_dataset)

# Save the processed dataset to disk
full_dataset.save_to_disk("/home/ubuntu/merged_dataset_to_disk")
