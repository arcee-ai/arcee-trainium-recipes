# arcee-trainium-recipes
The repository contains all the set-up required to execute trainium training jobs. 

## Data Preprocessing
For setting up Trainium CPT, it's crucial to have a tokenized and properly packed Hugging Face dataset that can be directly loaded from disk. In our experimental setup, we followed these steps:
1. **Text Extraction and Tokenization**:
   - Initially, we extracted the entire text from papers and tokenized them in a distributed setting.

2. **Packing Tokenized Data**:
   - We then packed the tokenized data in a format where each example contains 4096 tokens.
   - It's important to note that during the tokenization process, if the tokenized text exceeds 4096 tokens, it will be split into multiple tokens. Conversely, if it's fewer than 4096 tokens, it will be appended with other shorter examples.

3. **Saving the Dataset**:
   - Finally, we saved the dataset to disk using the Hugging Face dataset's library function `save_to_disk`.

These preprocessing steps ensure that the data is formatted correctly for use with Trainium CPT. For further details on the preprocessing pipeline, refer to the documentation or code provided.

## Model Training

First, convert a Hugging Face pretrained checkpoint to the NxD sharded format, and then, begin training with the pretrained weights. This example uses `meta-llama/Llama-2-7b-hf`. It is recommended running this code from `/fsx/` on your ParallelCluster. Example: we used /fsx/llama2 as our working directory.

To run this code:

* Clone the repository to `/fsx/llama2`.

* Update the `DATA_PATH` in `tp_zero1_llama2_7b_hf_pretrain.sh` to point to your tokenized dataset (tokenized and packed data resulted from the data preprocessing part).

* Make sure you have the `aws_neuron_venv_pytorch` virtualenv activated and install the dependecies by running the following commands:

```
python -m pip install neuronx_distributed --extra-index-url https://pip.repos.neuron.amazonaws.com
```

```
python3 -m pip install -r requirements.txt
```

* Run `huggingface-cli login` and enter your HF token, which will be required to download the `meta-llama/Llama-2-7b-hf` weights.

* Run `submit_ckpt_download_convert_job.sh`, which will launch a slurm job to download and convert the `meta-llama/Llama-2-7b-hf` checkpoint. The code for this job is in `download_and_convert_llama_ckpt.sh`. We recommend using a slurm job to run this as the head node of the cluster doesn’t have much RAM and will likely throw out of memory (OOM) error. When this job is done, you should see a directory named `llama2_7b_hf_sharded` in your working dir.

* Before training the model, you can set all the configurations in the `tp_zero1_llama2_7b_hf_pretrain.sh` file:
    You can set the [tensor parallelism](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/tensor_parallelism_overview.html#tensor-parallelism-overview) degree (`TP_DEGREE`), which should be a power of 2 and for better performance, choose the number of key value heads to be divisible by the TP size as the key and value projections should be sharded accross the TP ranks. 

    You can also update the global-batch size (`GBS`) (Larger GBS may result in smoother gradients and more stable updates to the model parameters, potentially leading to faster convergence), and mini-batch (`MBS`) (Increasing MBS helps with speeding up the training speed, but for now, only `MBS=1` is supported). 
    
    Depending on the number of epochs you plan to do training for, you have to calculate the total number of steps per epoch. For this, you can run `dp_steps_per_epoch.py` script after setting `dataset_path`, `global_batch_size`, `mini_batch_size`, `cores`, `nodes`, and `tensor_parallel_size` in line with the configurations in `tp_zero1_llama2_7b_hf_pretrain.sh` file. Then, you can update the `TOTAL_STEPS` parameter to the total number of steps per epoch multiplied the by total number of epochs. 
    
    We set `LR` equal to 1.5e-4 and `WARMUP_STEPS` equal to ~1% of the total number of steps per epoch as suggested in [this paper](https://openreview.net/pdf?id=pg7PUJe0Tl). Warm-up steps can help stabilize the training process by gradually introducing larger learning rates. If the warm-up period is too short, the learning rate may not have enough time to increase to an appropriate level, leading to underfitting. If the warm-up period is too long, the model may overfit to the training data during the initial stages of training. 
    
    Make sure that the `SEQ_LEN` matches the `max_position_embeddings` in `config.json` file. It should be equal to`4096` in this example. 
    
    `NUM_NEURONCORES` shows the total number of cores per Trainium node and `WORLD_SIZE` indicates the total number of nodes, which is automatically calculated by the bash file. 
    
    You can change `--logging_interval` to control per how many steps log the performance of the model. 
    
    For checkpointing, you can change the `--checkpoint_dir`, `--num_kept_checkpoint`, and `--checkpoint_freq` (in terms of the number of steps). 
    
    To enable training from a pretrained model, you can enable `--load_pretrained_checkpoint` and pass the pretrained checkpoint directory to `--pretrained_checkpoint_dir` argument.

* When the checkpoint download/conversion job is complete, run `submit_precompilation_job.sh` to precompile the graphs and populate the Neuron cache
when the precompilation job is complete, run `submit_training_job.sh` to launch the training job.

* When model training is done, you can convert it back from sharded format to Hugging Face executable format for inference and testing. For this, you can set the `input_dir` (directory of the sharded model), `output_dir` (target directory for the full model), and `tp_size` in the `convert_to_full_model.sh` file and then, execute `submit_convert_to_full.sh` bash file.


**Note:** You will need to change the number of nodes in the precompilation/training slurm launch scripts to match your cluster.

**Note:** Remember to delete the files under `neuron_compile_cache` and `__pycache__` folder before starting to train a new model.

**Note:** After submitting a job, you can track the logs in `slumr-XX.out` files, which will automatically appear under your working directory. To check if the nodes are idle or activate, you can use `sinfo` command and for more detailed information about the nodes, you can use `scontrol show nodes` and check the state of each node. When running is done, the nodes will return back to idle mode dynamically and except for the Head Node, you don't have to manually stop them. You can find the JOBID by running `squeue`. In order to stop a job, you can run `scancel <JOBID>`.


## Additional Resources
Reference 1: [Link](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/tutorials/training_llama2_7b.html) 

Reference 2: [Link](https://docs.google.com/document/d/1531RU5a9UnE3JNPC7R7iETLe5rdgBfdrIwrJoIQ0tWg/edit?usp=sharing)

