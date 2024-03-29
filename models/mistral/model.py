from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
from langchain.prompts import PromptTemplate
from IPython.display import display, Markdown
from langchain.chains import LLMChain
from transformers import BitsAndBytesConfig
from langchain.llms import HuggingFacePipeline
import json
import torch

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
        quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map="auto",,quantization_config=quantization_config)
        model.eval()

        return model
    
    def pipline(self):
        model = self.model()
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        print('tokenizer loaded')
        pipeline_inst = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            use_cache=True,
            device_map="auto",
            max_length=2500,
            do_sample=True,
            top_k=5,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
)
        print('pipeline loaded')
        llm = HuggingFacePipeline(pipeline=pipeline_inst)
        print('llm loaded')
        return llm
    

    def generate_response(self, input, label):
        template_c= '''
        You are given an example dataset before the transformation {input_list} and after the transformation {output_list}.

        Your goal is to generate a Python code that would reproduce the data transformation process,

        the code should be able to take in a different input dataset and perform the same data transformation steps.

        So don't use any specific example data inputs in the genetated code.

        No Explanation needed in your answer and your output should be of the following format:

        Generated Code:
        Your python code and comment here.

        End of code generation!

    '''
        template = """<s>[INST] You are an respectful and helpful assistant, 
        respond always be precise, assertive and politely answer in few words conversational english.
        Answer the question below from context below :""" + template_c + """[/INST] </s>
        """
        llm = self.pipline()
        prompt = PromptTemplate(template=template, input_variables=['input_list', 'output_list'])
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        response = llm_chain.run({'input_list': input, 'output_list': label})
        return response
    
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
    
    def gpt_output(self):
        result= []
        for j in range(2,3):
            if j in [31, 32, 35, 38, 39, 42,50]:
                continue
            else:
                for i in range(1,6):
                    path = '../../data/foofah/foofah/exp0_'+str(j) + '_'+ str(i)+ '.txt'
                    input_data, test_data = self.read_in_data(path)
                    #
                    tutorial = self.generate_response(input_data[0],input_data[1])
                    result.append([str(j) + '_'+ str(i), tutorial])
                    self.ans.append([str(j) + '_'+ str(i), tutorial])
                    print(j,i)
        return result
    


