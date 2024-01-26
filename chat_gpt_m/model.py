
import os
import openai
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
import sys
sys.path.append('../')
from data_access import read_in_data

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
    
    
    def gpt_output(self, template_c):
        result= []
        for j in range(2,3):
            for i in range(1,2):
                path = '../data/foofah/exp0_'+str(j) + '_'+ str(i)+ '.txt'
                input_data, test_data = read_in_data(path)
                llm = self.environemnt_setup()

                p_tutorial= PromptTemplate(input_variables=['input_list', 'output_list'],
                                    template=template_c)
                chain1 = LLMChain(llm = llm, prompt = p_tutorial)
                with get_openai_callback() as cb:
                    tutorial = chain1.run({'input_list': input_data[0], 'output_list': input_data[1]})
                    print(cb.total_tokens)
                
                result.append([str(j) + '_'+ str(i), tutorial])
                
        return input_data[0],input_data[1], chain1, result
    
    
    
