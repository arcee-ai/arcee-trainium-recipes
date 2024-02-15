from datasets import load_dataset
from itertools import chain
import os
import glob
import multiprocessing

dataset_path = "/home/ubuntu/parquet_dataset/"
save_path_base = "~/medical_packed_4096"
block_size = 4096
log_file = "failed_processes.log"  # Log file to record failures

def process_directory(directory):
    try:
        # To save each chunk separately.
        save_path = os.path.join(os.path.expanduser(save_path_base), os.path.basename(directory.rstrip('/')))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # parquet_files = glob.glob(f"{directory}/**/*.parquet", recursive=True)
        parquet_files = glob.glob(f"{directory}/*.parquet")

        # Load the tokenized data from parquet files
        tokenized_datasets = load_dataset('parquet', data_files=parquet_files).remove_columns(['token_count', 'text']).rename_column('tokens', 'input_ids')

        # Main data processing function
        def group_texts(examples):
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            concatenated_examples['attention_mask'] = [1] * len(concatenated_examples['input_ids'])
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            total_length = (total_length // block_size) * block_size
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            load_from_cache_file=True,
            desc=f"Grouping texts in chunks of {block_size} for the directory {os.path.basename(directory)}"
        )

        train_dataset = lm_datasets["train"]
        print(f"Processed {len(train_dataset)} items in {os.path.basename(directory)}")
        
        # Save the processed dataset to disk
        train_dataset.save_to_disk(save_path)

    except Exception as e:
        # Log the failure
        with open(log_file, "a") as log:
            log.write(f"Failed to process directory {directory}: {e}\n")

if __name__ == "__main__":
    directories = [d for d in glob.glob(dataset_path + "*/") if os.path.isdir(d)]
    with multiprocessing.Pool(processes=30) as pool:
        pool.map(process_directory, directories)

