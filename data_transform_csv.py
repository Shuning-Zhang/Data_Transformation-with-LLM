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