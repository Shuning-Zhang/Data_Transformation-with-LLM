from model import mistral_model
#import pandas as pd


mistral = mistral_model("mistralai/Mistral-7B-Instruct-v0.1")



result = mistral.gpt_output()

output= []

for x in test_test:
  code = x[1].split('[/INST] </s>')
  if "```python\n" in code:
        o = code[1].split("```python\n")
  else:
        o = code[1].split('Generated Code:\n')
  if len(o) == 2:
    d = {'data': x[0],'output':o[1]}
  else:
    d = {'data': x[0],'output':o[0]}

  # o = o[1].split("\n```\nEnd of code generation!")
  # d = {'data': x[0],'output':o[1]}
  print(d)

  output.append(d)
with open('output_mistral.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    
    # Write each row to the CSV file
    writer.writerows(output)

#df = pd.DataFrame(output)
#df.to_csv('output_mistral_2.csv', index=False)
