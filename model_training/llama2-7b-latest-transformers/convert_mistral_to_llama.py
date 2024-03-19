from transformers import LlamaForCausalLM, AutoTokenizer, LlamaConfig, AutoModelForCausalLM

#################### Llama Modification ###########################

# Initializing a LLaMA llama-7b style configuration
configuration = LlamaConfig()

# Modify the intermediate size
configuration.intermediate_size = 14336  # Set your desired value here

# Fuck it!! just modify the key_value heads :D :D 
configuration.num_key_value_heads = 8 

# Load the model with the updated configuration
llama_model = LlamaForCausalLM._from_config(config=configuration)

#################### Mistral ###########################

# Load the tokenizer from the Mistral model
mistral_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2") 

# Load the Mistral-7B model
mistral_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

# # Extract mistral weights
mistral_weights = mistral_model.state_dict()

# ######### Init and Push ###################

llama_model.load_state_dict(mistral_weights, strict=True)

llama_model.push_to_hub("arcee-ai/llama_from_mistral_instruct_v2", token="hf_****")
mistral_tokenizer.push_to_hub("arcee-ai/llama_from_mistral_instruct_v2", token="hf_*****")
