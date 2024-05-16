import numpy as np
import h5py
from utils import save_h5, load_h5, mask_array
import os
import matplotlib.pyplot as plt
from metric import IntersectionOverUnion, DiceIndex, clDice

#Indicate key
my_key = '101'

#Open predictions
save_name = my_key+".h5"
save_directory = '/group/dl4miacourse/The_Segmenters/Predictions/' + save_name
predictions = load_h5(save_directory)

#Print predictions type, shape, unique values and print an example
print(type(predictions))
print(predictions.shape)
np.unique(predictions)
plt.imshow(predictions[1,...])

#Print predictions histogram
plt.hist(predictions.flatten())

#Mask predictions and save results
mask_save_file_directory = "/group/dl4miacourse/The_Segmenters/Masks/"+my_key
mask_predictions = mask_array(predictions, threshold=0.5, mask_values=(1,0), output_dtype=np.uint8, saving_folder=mask_save_file_directory, saving_key=my_key)

#mask_predictions = mask_array(predictions, threshold=0.5)
print(mask_predictions.shape)
print(np.unique(mask_predictions))

#Open ground truth
gt = load_h5('/group/dl4miacourse/The_Segmenters/Data/test_label.h5')
print(type(gt))
print(gt.shape)
print(np.unique(gt))
print(gt.dtype)

#IoU Metric
metric = IntersectionOverUnion(mask_predictions, gt)
val_metr=metric.forward()
print(val_metr)
#print(metric)


#Dice Metric
metricDice = DiceIndex(mask_predictions, gt)
dice_val_metr=metricDice.forward()
print(dice_val_metr)
#print(metric)

#cldice Metric
metric_clDice = clDice(mask_predictions, gt)
print(metric_clDice)
#print(metric)