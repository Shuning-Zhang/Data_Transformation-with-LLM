
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
        You are given an example dataset before the transformation {input_list} and after the transformation {output_list}. 
        
        Your goal is to generate a Python code that would reproduce the data transformation process, 
        
        the code should be able to take in a different input dataset and perform the same data transformation steps. 
        
        You should not include Explanation in your answer and your output should be of the following format:

        Generated Code:
        Your python code and comment here.

    ''' 

# o stands for output
input, output,chain, result = GPT_Model.gpt_output(template_c)
print('input:',input)
print('output:',output)
print('chain:',chain)
# output= []
# for x in result:
#     if 'python\n' in x[1]:
#         o = x[1].split('python\n')
#     else:
#         o = x[1].split('Generated Code:\n\n')


#     o = o[1].split('\n\ninput_data')
#     d = {'data': x[0],'output': o[0]}

#     output.append(d)
# df = pd.DataFrame(output)
# df.to_csv('output_chat_gpt.csv', index=False)
