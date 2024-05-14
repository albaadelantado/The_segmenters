from utils import *
from tifffile import imread

datapath = '/group/dl4miacourse/The_Segmenters/Data/'
vol = load_h5(datapath + "ch-1_subv.h5")
label = imread(datapath + 'ch-1_subv_labkit_opened_np_cleaned.tiff')

# splitting to train, val, test h5 volumes
# 0.7 0.2 0.1
# 84 24 12, with two holes between test and val

# shape of patches will be 128 by 128
# we don't care about data leak into the val set for now, but we separate the test by 2 slices
# a corner is cut off along the x dimension, we use that for training
test = vol[:12,:,:]
val = vol[14:38,:,:]
train = vol[38:,:,:]

print(test.shape, val.shape, train.shape)

test_label = label[:12,:,:]
val_label = label[14:38,:,:]
train_label = label[38:,:,:]

arrays = [test,val,train,test_label, val_label, train_label]
names =  ['test','val','train','test_label', 'val_label', 'train_label']

for data, filename in zip(arrays,names):
    save_h5(data, datapath + filename)


