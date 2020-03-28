from model import unet, ModelCheckpoint
import numpy as np
from data import trainGeneratorArray, get_data
import matplotlib.pyplot as plt


def plot_samples(generator1, num_batches=1, save_path='demo_figs'):

    """

    Plot samples from a generator

    """

    cc = 0
    for im_sample, msk_sample in Gen1:
        implt = np.squeeze(im_sample[0, :, :, 0])
        mskplt = np.squeeze(msk_sample[0, :, :, 0])

        print("Maximum HU value: {0} and minimum HU value: {1}".format(np.max(implt), np.min(implt)))
        print("Number of classes: {0}".format(np.max(mskplt)))
        print("Image shape: {0}".format(implt.shape))

        f1 = plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(implt, cmap='gray')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(mskplt, vmax=3, vmin=0)
        plt.axis('off')
        plt.savefig(save_path + '/' + str(cc) + 'png')
        cc += 1
        if cc >= num_batches:
            break

def plot_samples_test(x, y, num_batches=1, save_path='demo_figs'):

    """

    Plot all cases from an array

    """

    im_len = x.shape[0]
    shp = y.shape

    for cc in range(im_len):
        implt = np.squeeze(x[cc, :, :, 0])

        # Reverse the one-hot-encoding by summing by index
        y1 = y[cc, :, :, :]
        mult = np.array(range(shp[-1]))
        mult = mult[np.newaxis, np.newaxis, :]
        y1 = y1 * mult
        mskplt = np.sum(y1, axis=-1)

        # Squeeze the mask

        print("Maximum HU value: {0} and minimum HU value: {1}".format(np.max(implt), np.min(implt)))
        print("Number of classes: {0}".format(np.max(mskplt)))
        print("Image shape: {0}".format(implt.shape))

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(implt, cmap='gray')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(mskplt, vmax=3, vmin=0)
        plt.axis('off')
        plt.savefig(save_path + '/' + str(cc) + 'pred.png')

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


# Model train

train_model = False

# Load the data
im, msk = get_data(tr_im_path, tr_msk_path)

# Build a generator      
Gen1 = trainGeneratorArray(2, im, msk, data_gen_args, save_to_dir = None)

# Plot samples from the generator
plot_samples(Gen1, num_batches=10)

model = unet()

if train_model:
    model_checkpoint = ModelCheckpoint('unet_covid.hdf5', monitor='loss',verbose=1, save_best_only=True)
    model.fit_generator(Gen1,steps_per_epoch=500,epochs=1,callbacks=[model_checkpoint])
else:
    model.load_weights('unet_covid.hdf5')

tst, _ = get_data(tst_im_path)
result = model.predict(tst)
result = result > 0.2

plot_samples_test(tst, result)


x=1

# testGene = testGenerator("data/membrane/test")
# results = model.predict_generator(testGene,30,verbose=1)
# saveResult("data/membrane/test",results)