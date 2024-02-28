
import os
import openai
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
from model import GPT_Model



# api key:
#3.5
# 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
#4.0
# sk-GardnCuVxSwbqswZ1a55F262A4D74431Af05AbD6D5C72dA2
# set up the gpt model

GPT_Model = GPT_Model('sk-GardnCuVxSwbqswZ1a55F262A4D74431Af05AbD6D5C72dA2',3.5)

#GPT_Model = GPT_Model('sk-GardnCuVxSwbqswZ1a55F262A4D74431Af05AbD6D5C72dA2',4.0)
# setting up template
template_c= '''

    Given an example input and output dataset, learn how the transformation performed from the provided input dataset to the output dataset. 
        
    Pay attention to the size difference between the input and output dataset.
    
    Generate python function with no explanations needed. 
        
    input dataset: {input_list} 
    output dataset: {output_list}



'''

# o stands for output
result = GPT_Model.gpt_output(template_c)
# print(result)
output= []
for x in result:
    o = x[1].split('\n\ninput_data')
    d = {'data': x[0],'output': o[0]}

    output.append(d)
df = pd.DataFrame(output)
df.to_csv('output_chat_gpt_3_5.csv', index=False)


