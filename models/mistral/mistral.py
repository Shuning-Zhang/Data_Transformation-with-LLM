from model import mistral_model
import pandas as pd


mistral = mistral_model("mistralai/Mistral-7B-Instruct-v0.1")



result = mistral.gpt_output(mistral)

output= []

for x in result:
  code = x[1].split('[/INST] </s>')
  # print(code[1])
  if "```python\n" in code:
        o = code[1].split("```python\n")
  else:
        o = code[1].split('Generated Code:\n')

  o = o[1].split("\n```\nEnd of code generation!")
  d = {'data': x[0],'output':o[0]}
  print(d)

  output.append(d)
df = pd.DataFrame(output)
df.to_csv('output_mistral_2.csv', index=False)