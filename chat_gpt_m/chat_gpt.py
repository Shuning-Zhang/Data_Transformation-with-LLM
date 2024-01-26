
import os
import openai
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
from model import GPT_Model



# api key:
# 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



# set up the gpt model
GPT_Model = GPT_Model('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')
# setting up template
template_c= '''

    Given an example input and output dataset, learn how the transformation performed from the provided input dataset to the output dataset. 
        
    Pay attention to the size difference between the input and output dataset.
    
    Generate python function with no explanations needed. 
        
    input dataset: {input_list} 
    output dataset: {output_list}



'''

# o stands for output
result = GPT_Model.gpt_output(template_c, GPT_Model)
# print(result)
output= []
for x in result:
    o = x[1].split('\n\ninput_data')
    d = {'data': x[0],'output': o[0]}

    output.append(d)
df = pd.DataFrame(output)
df.to_csv('output_chat_gpt_part.csv', index=False)
