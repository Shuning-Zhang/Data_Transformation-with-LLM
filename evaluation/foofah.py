import sys

sys.path.append("../")

from data_access import read_in_data, read_output_data


def calculate_acc(lower_bound=2, upper_bound=0):
    # Read in data
    acc = []
    for j in range(lower_bound, upper_bound + 1):
        if j == 9:
            continue
        for i in range(1, 6):
            path = "exp0_" + str(j) + "_" + str(i) + ".txt"
            input_data, test_data = read_in_data("../data/foofah/" + path)
            output_data = read_output_data(
                "../output/foofah/output_data_0_" + str(j) + "_" + str(i) + ".json"
            )

            # Calculate accuracy
            if output_data[0]:
                acc_temp = 0
                # print(j,i)
                for k in range(min(len(output_data[0]), len(test_data[1]))):
                    if output_data[0][k] == test_data[1][k]:
                        acc_temp += 1
                acc.append(acc_temp / max(len(output_data[0]), len(test_data[1])))
            else:
                acc.append(0)
    return sum(acc) / (len(acc) + 5)
