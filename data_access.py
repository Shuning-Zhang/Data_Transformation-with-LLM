import json

def read_in_data(file_name):
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


def read_output_data(file_name):
    test_data = None    
    output_data = ['']
    with open(file_name, 'rb') as f:
        output_data = json.load(f)


 

    return output_data


def read_llm_output_data(file_name):
    output_data = ['']
    with open(file_name, 'rb') as f:
        output_data = json.load(f)
 
    return output_data