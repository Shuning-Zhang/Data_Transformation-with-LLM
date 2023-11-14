import os
import openai
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
from data_transform_csv import read_in_data
# api key:
# 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


class GPT_Model:

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.total_token = 0
        self.ans = []

    

    def environemnt_setup(self):
        os.environ["OPENAI_API_KEY"] = self.api_key
        openai.api_key = os.environ["OPENAI_API_KEY"]
        llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature = 0.0)
        return llm
    
def gpt_output(template_c, GPT_Model):
    result= []
    for j in range(2,3):
        for i in range(1,2):
            path = 'data/exp0_'+str(j) + '_'+ str(i)+ '.txt'
            input_data, test_data = read_in_data(path)
            llm = GPT_Model.environemnt_setup()
            # tutorial = GPT_Model.get_tutorial(llm, input_data)

            p_tutorial= PromptTemplate(input_variables=['input_list', 'output_list'],
                                template=template_c)
            chain1 = LLMChain(llm = llm, prompt = p_tutorial)
            with get_openai_callback() as cb:
                tutorial = chain1.run({'input_list': input_data[0], 'output_list': input_data[1]})
                print(cb.total_tokens)
            
            result.append([str(j) + '_'+ str(i), tutorial])
            
            # display(Markdown(f"<b>{tutorial}</b>"))
    return result


# set up the gpt model
GPT_Model = GPT_Model('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')
# setting up template
template_c= '''
        You are given an example dataset before the transformation {input_list} and after the transformation {output_list}. 
        
        Your goal is to generate a Python code that would reproduce the data transformation process, 
        
        the code should be able to take in a different input dataset and perform the same data transformation steps. 
        
        You should not include Explanation in your answer and your output should be of the following format:

        Generated Code:
        Your python code and comment here.

    ''' 

# o stands for output
result = gpt_output(template_c, GPT_Model)
output= []
for x in result:
    if 'python\n' in x[1]:
        o = x[1].split('python\n')
    else:
        o = x[1].split('Generated Code:\n\n')


    o = o[1].split('\n\ninput_data')
    d = {'data': x[0],'output': o[0]}

    output.append(d)
df = pd.DataFrame(output)
df.to_csv('output_chat_gpt.csv', index=False)
