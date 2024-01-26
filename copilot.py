# The function will perform data transformation and return a python list.
# The rule of the data transformation is the same as the following example:
# input_list:
# [["3099", "905", " AUST 4WD CUST ACT", "", "", "", ""], 
#  ["NO.14", "NO.14", "Full Copies", "6.7839", "2", "* *", "0"]]
# output_list:
#[["3099", "905", " AUST 4WD CUST ACT", "NO.14", "Full Copies", "6.7839", "2"]]
#
#the function should group every other line of the input list together and remove the empty rows, na elements within a sublist and duplicates.
# the function should also remove any non alphanumeric characters from the input list.


import pandas as pd
from data_access import read_in_data
input_data, test_data = read_in_data('data/exp0_2_1.txt')
    
def data_transformation(inputdata):
    
    #remove empty rows
    input_data = [x for x in inputdata if x != []]

    #remove na elements within a sublist 
    for i in range(len(input_data)):
        input_data[i] = [x for x in input_data[i] if x != 'NA']

    #remove duplicates
    for i in range(len(input_data)):
        input_data[i] = list(set(input_data[i]))
    
    output_data = []
    for i in range(0, len(input_data), 2):
        output_data.append([input_data[i] + input_data[i+1]])
    
    return output_data

