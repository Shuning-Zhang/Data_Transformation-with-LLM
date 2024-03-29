from model import mistral_model
import pandas as pd


mistral = mistral_model("mistralai/Mistral-7B-Instruct-v0.1")

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
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible for the user. Don't provide info you weren't asked to provide."""


llm = mistral.pipline()

result = mistral.gpt_output(llm,template_c, sys_prompt)

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
df.to_csv('output_mistral_2.csv', index=False)