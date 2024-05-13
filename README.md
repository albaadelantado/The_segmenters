# Hello world!

We will be working on a lightsheet volume of human brain with fluorescent stain for vasculature.

The labeled ground truth and bigger raw volume are here:

'/group/dl4miacourse/The_Segmenters/Data'

This folder contains: 

- ch-1_subv.h5 = labeled subvolume

- ch-1_subv_labkit_opened_np_cleaned.tiff = labels, first generated with labkit then refined manually in napari

- uni_tp-0_ch-1_st-0-x00-y00_obj-right_cam-right_etc.lux.h5 = full raw 3 gb volume



## TODO

first, 2D unet
then decide how we want to do 3D (3D unet? cellpose like?)
copy paste everything


- split train val test - 10 test 20 val 70 train along x
- general dataset (with transformations) 
    - then into pytorch dataloader
- pick baseline model architecture
- pick loss (https://github.com/jocpae/clDice ?)
- write training loop 
- and validation loop
- define metrics
- refine results

- add denoising

