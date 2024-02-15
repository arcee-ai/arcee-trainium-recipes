# arcee-trainium-recipes
The repository contains all the set-up required to execute trainium training jobs. 

## Data Preprocessing

## Model Training

First, convert a Hugging Face pretrained checkpoint to the NxD sharded format, and then begin training with the pretrained weights. This example uses `meta-llama/Llama-2-7b-hf`. It is recommended running this code from `/fsx/` on your ParallelCluster. Example: we used /fsx/llama2 as our working directory.

To run this code:

* Clone the repository to `/fsx/llama2`.

* Update the `DATA_PATH` in `tp_zero1_llama2_7b_hf_pretrain.sh` to point to your tokenized dataset (tokenized and packed data resulted from the data preprocessing part).

* Make sure you have the `aws_neuron_venv_pytorch` virtualenv activated and install the dependecies using the requirements.txt file.

* Run `huggingface-cli login` and enter your HF token, which will be required to download the `meta-llama/Llama-2-7b-hf` weights.

* Run `submit_ckpt_download_convert_job.sh`, which will launch a slurm job to download and convert the `meta-llama/Llama-2-7b-hf` checkpoint. The code for this job is in `download_and_convert_llama_ckpt.sh`. We recommend using a slurm job to run this as the head node of the cluster doesn’t have much RAM and will likely throw out of memory (OOM) error. When this job is done, you should see a directory named `llama2_7b_hf_sharded` in your working dir.

* When the checkpoint Download/Conversion job is complete, run `submit_precompilation_job.sh` to precompile the graphs and populate the Neuron cache
when the precompilation job is complete, run `submit_training_job.sh` to launch the training job.


**Note:** You will need to change the number of nodes in the precompilation/training slurm launch scripts to match your cluster.


## Additional Resources

