from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
from langchain.prompts import PromptTemplate
from IPython.display import display, Markdown
from langchain.chains import LLMChain
from langchain.llms import HuggingFacePipeline
import json


#model_id = "../../Mistral-7B"
# model_id = "mistralai/Mistral-7B-Instruct-v0.1"
# tokenizer = AutoTokenizer.from_pretrained(model_id)

# model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

# messages = [
#     {"role": "user", "content": "What is your favourite condiment?"},
#     {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
#     {"role": "user", "content": "Do you have mayonnaise recipes?"}
# ]

# inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

# outputs = model.generate(inputs, max_new_tokens=20)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))


class mistral_model:
    DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible for the user. Don't provide info you weren't asked to provide."""

    def __init__(self, model: str) -> None:
        self.model_id = model
        self.ans = []

    def model(self):
        
        model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map="auto")
        model.eval()

        return model
    
    def pipline(self):
        model = self.model()
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        print('tokenizer loaded')
        generate_text = pipeline(
            model=model, tokenizer=tokenizer,
            return_full_text=True,  # langchain expects the full text
            task='text-generation',
            # we pass model parameters here too
            temperature=0.05,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
            max_new_tokens=512,  # mex number of tokens to generate in the output
            repetition_penalty=1.1  # without this output begins repeating
            )
        print('pipeline loaded')
        llm = HuggingFacePipeline(pipeline=generate_text)
        print('llm loaded')
        return llm
    

    def get_prompt(self, instruction, system_prompt = DEFAULT_SYSTEM_PROMPT):
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        
        SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS
        prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
        return prompt_template
    
    def read_in_data(self,file_name):
        test_data = None
        input_data = [''] * 2
        output_data = [''] * 2
        with open(file_name, 'rb') as f:
            test_data = json.load(f)

        input_data[0] = test_data['InputTable']
        input_data[1] = test_data['OutputTable']
        output_data[0] = test_data['TestingTable']
        output_data[1] = test_data['TestAnswer']


        return input_data, output_data
    
    def gpt_output(self, llm, template_c, sys_prompt):
        result= []
        for j in range(2,3):
            if j in [31, 32, 35, 38, 39, 42,50]:
                continue
            else:
                for i in range(1,6):
                    path = '../data/foofah/foofah/exp0_'+j + '_'+ str(i)+ '.txt'
                    input_data, test_data = self.read_in_data(path)
                    #
                    prompt_template = self.get_prompt(template_c,sys_prompt)
                    p_tutorial= PromptTemplate(template=prompt_template,input_variables=['input_list', 'output_list'])
                    chain1 = LLMChain(llm = llm, prompt = p_tutorial)

                    tutorial = chain1.run({'input_list': input_data[0], 'output_list': input_data[1]})
                    result.append([str(j) + '_'+ str(i), tutorial])
                    self.ans.append([str(j) + '_'+ str(i), tutorial])
                    print(j,i)
        return result
    


