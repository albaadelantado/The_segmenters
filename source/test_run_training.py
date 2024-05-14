import dataset
import os

train_data_path = "train"

train_data = dataset.AngioDataset(data_path)
data_1_raw = data[0][0]
data_1_mask = data[0][1]

print(type(data_1_raw))
print(data_1_raw.shape)

