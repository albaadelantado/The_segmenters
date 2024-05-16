import numpy as np
from utils import save_h5


def mask_array(array2mask, threshold=0.5, mask_values=(1,0), output_dtype=np.uint8, saving_folder="", saving_key="predictions"):
    masked_array = np.where(array2mask>threshold, mask_values[0],mask_values[1]).astype(output_dtype)
    if len(saving_folder)>0:
        save_h5(masked_array,saving_folder, key=saving_key)
    return masked_array
    

