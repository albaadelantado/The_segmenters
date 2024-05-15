import numpy as np
import h5py
import matplotlib.pyplot as plt
import torch
import os




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



def get_patch(data, patch_idx, patch_size, min_pad_ratio = 0.25, return_slice = False): 

    smallest_portion_dim = [s % d for s,d in zip(data.shape, patch_size)]
 
    ratios = np.divide(smallest_portion_dim,patch_size)
    ratios = np.array([x+1 for x in ratios if x == 0])
    assert np.all(ratios > min_pad_ratio) , "The edge patch contains too much padding. Change patch size"
        
    # getting indices to slice data, from size*idx to size*(idx+1) along all dims
    init_i = np.multiply(patch_idx,patch_size)
    end_i = np.add(init_i,patch_size)


    patch_npslice = np.s_[init_i[0]:end_i[0],init_i[1]:end_i[1],init_i[2]:end_i[2]]
    # slicing to get our data, will be smaller than the patch when we go over the edges
    portion = data[patch_npslice]

    # np.pad wants tuples for each axis (num_pad_elements_before, after)
    padding_size = list(zip(np.zeros(len(patch_size)).astype(int),np.subtract(patch_size,np.array(portion.shape))))
    padded = np.pad(portion,padding_size)
    
    if return_slice == True:
        return padded, patch_npslice
    else:
        return padded


def get_n_patch_per_dim(vol_shape,patch_size):
    return np.ceil(np.array(vol_shape) / np.array(patch_size)).astype(int)


def repatch(patch_list,patch_slice_list, vol_shape):
    init = np.zeros(vol_shape)

    for patch,npslice in zip(patch_list, patch_slice_list):
        init[npslice] = patch

    return init

    

def show_random_dataset_image_with_prediction(dataset, model, device="cpu"):
    idx = np.random.randint(0, len(dataset))  # take a random sample
    img, mask = dataset[idx]  # get the image and the nuclei masks
    x = img.to(device).unsqueeze(0)
    y = model(x)[0].detach().cpu().numpy()
    print("MSE loss:", np.mean((mask[0].numpy() - y[0]) ** 2))
    f, axarr = plt.subplots(1, 3)  # make two plots on one figure
    axarr[0].imshow(img[0])  # show the image
    axarr[0].set_title("Image")
    axarr[1].imshow(mask[0], interpolation=None)  # show the masks
    axarr[1].set_title("Mask")
    axarr[2].imshow(y[0], interpolation=None)  # show the prediction
    axarr[2].set_title("Prediction")
    _ = [ax.axis("off") for ax in axarr]  # remove the axes
    print("Image size is %s" % {img[0].shape})
    plt.show()



def save_checkpoint(model, optimizer, n_epoch, path, key="checkpoint"):
    save_path = os.path.join(path, f"{key}.pt")
    torch.save(
        {
            "model":model.state_dict(),
            "optimizer":optimizer.state_dict(),
            "epoch": n_epoch
        },
        save_path
    )


def load_checkpoint(model, path, optimizer=None, key="checkpoint"):
    load_path = os.path.join(path, f"{key}.pt")
    checkpoint=torch.load(load_path)
    model.load_state_dict(checkpoint["model"])

    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch=checkpoint["epoch"]
        return model, optimizer, epoch
    
    return model
