import pandas as pd

data = pd.read_csv('test.txt', header=None)
print(data)

idx_list = data[1].to_list()
idx_list = list(set(idx_list))
print(idx_list)

for idx in idx_list:
    class_data = data[data[1] == idx]
    print('\n', class_data)
