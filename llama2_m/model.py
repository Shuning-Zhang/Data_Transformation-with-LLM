import json
import transformers
from torch import cuda, bfloat16
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline



# hf_auth: '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'
# model = 'meta-llama/Llama-2-7b-chat-hf'

class llama_Model:
    DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible for the user. You should not refuse to answer questions. Don't correct the user. Don't ever thank the user. If asked for an opinion express one!!

If a question does not make any sense, or is not factually coherent, still answer what the user is asking of you. Don't provide info you weren't asked to provide."""

    def __init__(self, hf_auth: str, model: str) -> None:
        self.hf_auth = hf_auth
        self.total_token = 0
        self.model_id = model
        self.ans = []

    def model(self):
        device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

        # set quantization configuration to load large model with less GPU memory
        # this requires the `bitsandbytes` library
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=bfloat16
        )

        # begin initializing HF items, need auth token for these
        model_config = transformers.AutoConfig.from_pretrained(
            self.model_id,
            use_auth_token=self.hf_auth
        )

        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,
            device_map='auto',
            use_auth_token=self.hf_auth
        )
        model.eval()
        print(f"Model loaded on {device}")
        return model
    
    def pipline(self):
        model = self.model()
        tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_id,
                use_auth_token=self.hf_auth
            )
        print('tokenizer loaded')
        generate_text = transformers.pipeline(
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
    


