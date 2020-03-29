
import numpy as np
import matplotlib.pyplot as plt


def plot_samples(Gen1, num_batches=1, save_path='demo_figs'):

    """

    Plot samples from a generator

    """

    cc = 0
    for im_sample, msk_sample in Gen1:
        implt = np.squeeze(im_sample[0, :, :, 0])
        msk_sample = msk_sample[0]

        # Reverse the one-hot-encoding by summing by index
        mult = np.array(range(msk_sample.shape[-1]))
        mult = mult[np.newaxis, np.newaxis, :]
        msk_sample = msk_sample * mult
        msk_sample = np.sum(msk_sample, axis=-1)
        mskplt = np.squeeze(msk_sample)

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

def plot_samples_test(x, y, gt=None, num_batches=1, save_path='demo_figs', save_term=''):

    """

    Plot all cases from an array

    """

    def reverse_one_hot_encoding(y1):
        mult = np.array(range(shp[-1]))
        mult = mult[np.newaxis, np.newaxis, :]
        y1 = y1 * mult
        mskplt = np.sum(y1, axis=-1)
        return mskplt

    im_len = x.shape[0]
    shp = y.shape

    for cc in range(im_len):
        implt = np.squeeze(x[cc, :, :, 0])

        # Reverse the one-hot-encoding by summing by index

        y1 = y[cc, :, :, :]
        mskplt = reverse_one_hot_encoding(y1)
        mskplt = mskplt.transpose()
        mskplt_mask = np.ma.masked_array(mskplt, mskplt == 0)

        if gt is not None:
            gt1 = gt[cc, :, :, :]
            mskpltgt = reverse_one_hot_encoding(gt1)
            mskpltgt = mskpltgt.transpose()
            mskpltgt_mask = np.ma.masked_array(mskpltgt, mskpltgt == 0)


        implt = implt.transpose()

        fig = plt.figure(figsize=(20,10))
        fig.set_tight_layout(True)
        plt.subplot(1, 3, 1)
        plt.imshow(implt, cmap='gray')
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(implt, cmap='gray')
        plt.imshow(mskplt_mask, vmax=3, vmin=0, cmap='jet', alpha=0.5)
        plt.axis('off')
        if gt is not None:
            plt.subplot(1, 3, 3)
            plt.imshow(implt, cmap='gray')
            plt.imshow(mskpltgt_mask, vmax=3, vmin=0, cmap='jet', alpha=0.5)
            plt.axis('off')

        plt.savefig(save_path + '/' + str(cc) + save_term + '.png')
