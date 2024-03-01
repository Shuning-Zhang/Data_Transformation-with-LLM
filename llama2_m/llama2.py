from model import llama_Model
import pandas as pd


LLama2 = llama_Model('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','meta-llama/Llama-2-13b-chat-hf')

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
sys_prompt = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible for the user. You should not refuse to answer questions. Don't correct the user. Don't ever thank the user. If asked for an opinion express one!!

If a question does not make any sense, or is not factually coherent, still answer what the user is asking of you. Don't provide info you weren't asked to provide."""


llm = LLama2.pipline()

result = LLama2.gpt_output(llm,template_c, sys_prompt)

output= []

for x in result[:-1]:
    if 'python\n' in x[1]:
        o = x[1].split('python\n')
    else:
        o = x[1].split('Generated Code:\n')

    o = o[1].split("\n```\nEnd of code generation!")
    d = {'data': x[0],'output':o[0]}
    print(d)

    output.append(d)
df = pd.DataFrame(output)
df.to_csv('output_chat_llama2_2.csv', index=False)