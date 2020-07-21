import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from model import unet, ModelCheckpoint
import numpy as np
from data import trainGeneratorArray, get_data
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from plotting import plot_samples, plot_samples_test

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=False,
                    fill_mode='nearest')

# Paths to training data
tr_im_path = 'data/covid/tr_im.nii.gz'
tr_msk_path = 'data/covid/tr_mask.nii.gz'
tst_im_path = 'data/covid/val_im.nii.gz'


# Options 
# --------------------------------------

train_model = False
validation_size = 0.1
steps_per_epoch = 100
epochs = 200

# --------------------------------------

# Load the data
im, msk = get_data(tr_im_path, tr_msk_path)

imtr, imval, msktr, mskval = train_test_split(im, msk, test_size=validation_size, random_state=42)

# Build a generator      
Gen1 = trainGeneratorArray(2, imtr, msktr, data_gen_args, save_to_dir = None)

# Plot samples from the generator
# plot_samples(Gen1, num_batches=10)

model = unet()

if train_model:
    model_checkpoint = ModelCheckpoint('unet_covid_crossval_200.hdf5', monitor='loss',verbose=1, save_best_only=True)
    model.fit_generator(Gen1, 
                        steps_per_epoch=steps_per_epoch, 
                        epochs=epochs, callbacks=[model_checkpoint], 
                        validation_data=(imval, mskval))
else:

    model_name = 'crossval_200'
    model.load_weights('unet_covid_' + model_name + '.hdf5')

    # Test set

    imtst, _ = get_data(tst_im_path)
    imtstpred = model.predict(imtst)
    # imtstpred = imtstpred > 0.5
    plot_samples_test(imtst, imtstpred, save_term='_testset_' + model_name)

    # Validation set

    imvalpred = model.predict(imval)
    # imvalpred = imvalpred > 0.5
    plot_samples_test(imval, imvalpred, mskval, save_term='_valset_' + model_name)

