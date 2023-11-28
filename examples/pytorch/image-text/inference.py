from transformers import FuyuProcessor, FuyuForCausalLM
from PIL import Image
import requests
import torch

# load model and processor
model_id = "adept/fuyu-8b"
processor = FuyuProcessor.from_pretrained(model_id)

model = FuyuForCausalLM.from_pretrained(model_id,
                                        device_map="cuda:0",
                                        load_in_4bit=True,
                                        output_hidden_states=True,
                                        bnb_4bit_use_double_quant=True,
                                        bnb_4bit_quant_type="nf4",
                                        bnb_4bit_compute_dtype=torch.bfloat16)

# prepare inputs for the model
text_prompt = "What is 1+1?\n"
url = "https://huggingface.co/adept/fuyu-8b/resolve/main/bus.png"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=text_prompt, images=image, return_tensors="pt").to("cuda:0")

# print("config", model.config)

# autoregressively generate text
generation_output = model.generate(**inputs, max_new_tokens=1000, return_dict_in_generate=True)

output = generation_output['sequences']
hidden_states = generation_output['hidden_states']

print("last hidden state", hidden_states[-1][-1].size())

generation_text = processor.batch_decode(output[:, :], skip_special_tokens=True)[0]
print(generation_text[generation_text.index(text_prompt)+len(text_prompt):])
