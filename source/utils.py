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