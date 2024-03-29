import torch
from transformers import BitsAndBytesConfig
from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

tokenizer =transformers.AutoTokenizer.from_pretrained("../../Mistral-7B")
model = AutoModelForCausalLM.from_pretrained(
    "/content/Mistral-7B-v0.1/",
    trust_remote_code=True,
    device_map="auto",
    quantization_config=quantization_config,
    max_memory={0:"15GB"}
)

model.eval()
model = torch.compile(model, mode = "max-autotune", backend="inductor")

text = "What is the capital of India?"

device = 'cuda'
model_inputs = tokenizer(text, return_tensors="pt").to(model.device)

generated_ids = model.generate(**model_inputs, temperature=0.1, top_k=1, top_p=1.0, repetition_penalty=1.4, min_new_tokens=16, max_new_tokens=128, do_sample=True)
decoded = tokenizer.decode(generated_ids[0])
print(decoded)
