
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


# prose data test
def prose_test(GPT_Model, template_c):
    folder_path = "../data/foofah/Transformation.Text/"
    result = []
    visited_file = []
    for sub_file in os.listdir(folder_path):
        if sub_file != '.DS_Store':
            print(sub_file)
            visited_file.append(sub_file)
            file_path = os.path.join(folder_path, sub_file)
            input_data, output_data = GPT_Model.read_in_data(file_path)
            tutorial = GPT_Model.get_prose_output(input_data, template_c)
            result.append([sub_file, tutorial])
            GPT_Model.ans.append([sub_file, tutorial])
    return result, visited_file

# foofah data test
def foofah_test(GPT_Model, template_c):
    result = GPT_Model.gpt_output(template_c)
    return result



def main():
    # define template
    template_c= '''

    Given an example input and output dataset, learn how the transformation performed from the provided input dataset to the output dataset. 
        
    Pay attention to the size difference between the input and output dataset.
    
    Generate python function with no explanations needed. 
        
    input dataset: {input_list} 
    output dataset: {output_list}
    
     '''
    
    #define model 

    GPT_Model = GPT_Model('sk-GardnCuVxSwbqswZ1a55F262A4D74431Af05AbD6D5C72dA2',3.5)
    #GPT_Model = GPT_Model('sk-GardnCuVxSwbqswZ1a55F262A4D74431Af05AbD6D5C72dA2',4.0)

    # call tests on datasets
    result = prose_test(GPT_Model,template_c)
    #result = fooafah_test(GPT_Model,template_c)

    # convert result into csv
    output= []
    for x in result:
        o = x[1].split('\n\ninput_data')
        d = {'data': x[0],'output': o[0]}

        output.append(d)
    df = pd.DataFrame(output)
    df.to_csv('output_chat_gpt_3_5.csv', index=False)

main()
