import numpy as np
import h5py


def load_h5(path, key = None):

    with h5py.File(path) as f:
        if key == None:
            key = list(f.keys())[0]
        d = f[key]
        data = np.array(d)
    return data


def save_h5(data,filepath,key = 'data', dtype = 'uint16'):
    if filepath[-3:] != '.h5':
        # append extension
        filepath = filepath + '.h5'

    with h5py.File(filepath, "w") as f:
        dset = f.create_dataset(key, data.shape, dtype=dtype)
        dset.write_direct(data)
    print('Done writing to h5')


def crop(x, y):
        """Center-crop x to match spatial dimensions given by y."""

        x_target_size = x.size()[:-2] + y.size()[-2:]

        offset = tuple((a - b) // 2 for a, b in zip(x.size(), x_target_size))

        slices = tuple(slice(o, o + s) for o, s in zip(offset, x_target_size))

        return x[slices]



def get_patch(data, patch_idx, patch_size, min_pad_ratio = 0.25): 

    smallest_portion_dim = [s % d for s,d in zip(data.shape, patch_size)]
 
    ratios = np.divide(smallest_portion_dim,patch_size)
    ratios = np.array([x+1 for x in ratios if x == 0])
    assert np.all(ratios > min_pad_ratio) , "The edge patch contains too much padding. Change patch size"
        
    # getting indices to slice data, from size*idx to size*(idx+1) along all dims
    init_i = np.multiply(patch_idx,patch_size)
    end_i = np.multiply(np.add(patch_idx,1),patch_size)

    # slicing to get our data, will be smaller than the patch when we go over the edges
    portion = data[init_i[0]:end_i[0],init_i[1]:end_i[1],init_i[2]:end_i[2]]

    # np.pad wants tuples for each axis (num_pad_elements_before, after)
    padding_size = list(zip(np.zeros(len(patch_size)).astype(int),np.subtract(patch_size,np.array(portion.shape))))
    padded = np.pad(portion,padding_size)
    
    return padded




