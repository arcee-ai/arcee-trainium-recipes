import transformers
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer
import torch


config = transformers.AutoConfig.from_pretrained("./config.json") #Llama-2 config
model = transformers.AutoModelForCausalLM.from_config(config)

model.load_state_dict(torch.load("./ckpt_pt/step_1000/checkpoint.pt"))

print(model)

model.push_to_hub(repo_id="arcee-ai/no_zero_1_step_1000", token="hf_xxxxxSIeVFszzfCMxxxOtHT")

#model.save_pretrained("./hf_ckpt", safe_serialization=True) 