import sys

sys.path.append("../")
from data_access import read_in_data, read_llm_output_data


def calculate_acc(llm_model, lower_bound=2, upper_bound=8):
    acc = []
    for j in range(lower_bound, upper_bound):
        for i in range(1, 6):
            path = "../data/foofah/exp0_" + str(j) + "_" + str(i) + ".txt"
            input_data, test_data = read_in_data(path)
            output_data = read_llm_output_data(
                "../output/"
                + llm_model
                + "/output_data_0_"
                + str(j)
                + "_"
                + str(i)
                + ".json"
            )

            # compare output_data and test_data
            acc_temp = 0
            # print(j,i)
            for k in range(min(len(output_data), len(test_data[1]))):
                for m in range(min(len(output_data[k]), len(test_data[1][k]))):
                    if output_data[k][m] == test_data[1][k][m]:
                        acc_temp += 1
            acc.append(acc_temp / (len(test_data[1]) * len(test_data[1][0])))
    return acc
