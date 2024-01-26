import json
import transformers
import pandas as pd
from torch import cuda, bfloat16
from langchain.chains import LLMChain
from data_access import read_in_data
from langchain.prompts import PromptTemplate
from IPython.display import display, Markdown
from langchain.llms import HuggingFacePipeline

def setup(model_id: str):
    #model_id = 'meta-llama/Llama-2-7b-chat-hf'

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
    hf_auth = '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'
    model_config = transformers.AutoConfig.from_pretrained(
        model_id,
        use_auth_token=hf_auth
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
        use_auth_token=hf_auth
    )
    model.eval()
    print(f"Model loaded on {device}")
    return model

def token(model, model_id: str):

    hf_auth = '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'
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        use_auth_token=hf_auth
    )

    generate_text = transformers.pipeline(
        model=model, tokenizer=tokenizer,
        return_full_text=True,  # langchain expects the full text
        task='text-generation',
        # we pass model parameters here too
        temperature=0.0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        max_new_tokens=512,  # mex number of tokens to generate in the output
        repetition_penalty=1.1  # without this output begins repeating
    )
    llm = HuggingFacePipeline(pipeline=generate_text)

    return llm

def get_prompt(instruction, system_prompt):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

def llama_output(llm):
    template_c= '''
        You are given an example dataset before the transformation {input_list} and after the transformation {output_list}.

        Your goal is to generate a Python code that would reproduce the data transformation process,

        the code should be able to take in a different input dataset and perform the same data transformation steps.

        So don't use any specific example data inputs in the genetated code.

        You should not include Explanation in your answer and your output should be of the following format:

        Generated Code:
        Your python code and comment here.

        End of code generation!

    '''
    sys_prompt = """\
    You are a helpful, respectful and honest code generateing assistant. Always answer as helpfully as possible for the user. Don't correct the user. Don't ever thank the user. If asked for an opinion express one!!

    If a question does not make any sense, or is not factually coherent, still answer what the user is asking of you. Don't provide info you weren't asked to provide."""

    prompt_template = get_prompt(template_c,sys_prompt)
    p_tutorial= PromptTemplate(template=prompt_template,input_variables=['input_list', 'output_list'])
    chain1 = LLMChain(llm = llm, prompt = p_tutorial)

    result= []
    for j in range(2,6):
        for i in range(1,6):
            path = 'data/exp0_'+str(j) + '_'+ str(i)+ '.txt' #path to dataset
            input_data, test_data = read_in_data(path)
            # tutorial = GPT_Model.get_tutorial(llm, input_data)
            tutorial = chain1.run({'input_list': input_data[0], 'output_list': input_data[1]})
            result.append([str(j) + '_'+ str(i), tutorial])
    return result


model = setup('meta-llama/Llama-2-7b-chat-hf')
llm = token(model, 'meta-llama/Llama-2-7b-chat-hf')
result = llama_output(llm)
output= []

for x in result:
    if 'python\n' in x[1]:
        o = x[1].split('python\n')
    else:
        o = x[1].split('Generated Code:\n')


    o = o[1].split("\n```\nEnd of code generation!")
    d = {'data': x[0],'output': o[0]}
    print(d)

    output.append(d)
df = pd.DataFrame(output)
df.to_csv('output_chat_llama2.csv', index=False)