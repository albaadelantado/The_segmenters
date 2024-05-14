val_filepath = "val"
val_dataset = AngioDataset(val_filepath, patch_size=[1, 128, 128])
print(len(val_dataset))


imgs = []
slices = []
for i in range(len(val_dataset)):
    img, mask, npslice = val_dataset[i]
    imgs.append(img)
    slices.append(npslice)

vol = repatch(imgs, slices, val_dataset.vol.shape)