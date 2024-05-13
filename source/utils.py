import numpy as np
import h5py


def load_h5(path, key):
    with h5py.File(path) as f:
        d = f[key]
        data = np.array(d)
    return data


def save_h5():
    pass

