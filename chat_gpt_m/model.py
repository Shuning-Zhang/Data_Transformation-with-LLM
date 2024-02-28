
import os
import openai
import json
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
import sys
sys.path.append('../')
# from ..data_access import read_in_data


class GPT_Model:

    def __init__(self, api_key: str, model: int) -> None:
        self.api_key = api_key
        self.total_token = 0
        self.model = model
        self.ans = []

    

    def environemnt_setup(self):
        os.environ["OPENAI_API_KEY"] = self.api_key
        openai.api_key = os.environ["OPENAI_API_KEY"]
        if self.model == 3.5:
            openai.api_base = "https://oneapi.xty.app/v1"
            llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature = 0.0)
        elif self.model == 4.0:
            openai.api_base = "https://oneapi.xty.app/v1"
            llm = ChatOpenAI(model_name='gpt-4', temperature = 0.0)
        return llm
    
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
    
    
    def gpt_output(self, template_c):
        result= []
        
        name_list = ["craigslist_data_wrangler", "crime_data_wrangler", "potters_wheel_divide", "potters_wheel_fold" ,
                     "potters_wheel_fold_2", "potters_wheel_merge_split", "potters_wheel_split_fold", "potters_wheel_unfold", 
                     "potters_wheel_unfold2", "proactive_wrangling_fold", "proactive_wrangling_complex", "reshape_table_structure_data_wrangler"]
        for j in name_list:
            if j in [25, 31, 32, 35, 38, 39, 42,50]:
                continue
            for i in range(1,6):
                path = '../data/foofah/foofah/exp0_'+j + '_'+ str(i)+ '.txt'
                input_data, test_data = self.read_in_data(path)
                llm = self.environemnt_setup()

                p_tutorial= PromptTemplate(input_variables=['input_list', 'output_list'],
                                    template=template_c)
                chain1 = LLMChain(llm = llm, prompt = p_tutorial)
                with get_openai_callback() as cb:
                    tutorial = chain1.run({'input_list': input_data[0], 'output_list': input_data[1]})
                    print(j,i, ":",cb.total_tokens)

                    
                result.append([str(j) + '_'+ str(i), tutorial])
                self.ans.append([str(j) + '_'+ str(i), tutorial])
 
        # path = '../data/foofah/Transformation.Text/Address.000002.json'
        # input_data, test_data = read_in_data(path)
        # llm = self.environemnt_setup()

        # p_tutorial= PromptTemplate(input_variables=['input_list', 'output_list'],
        #                             template=template_c)
        # chain1 = LLMChain(llm = llm, prompt = p_tutorial)
        # with get_openai_callback() as cb:
        #     tutorial = chain1.run({'input_list': input_data[0], 'output_list': input_data[1]})
        #     print(cb.total_tokens)
                
        # result.append(tutorial)
                
        return result
    
    
    
