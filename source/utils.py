import numpy as np
import h5py


def load_h5(path, key):
    f = h5py.File(path)
    print(list(f.keys()))
    d = f[key]
    data = np.array(d)
    return data


def save_h5():
    pass

