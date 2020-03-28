from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
import nibabel as nib
from keras.utils import to_categorical
from tensorflow import one_hot
import tensorflow as tf


Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])


# def get_one_hot(n_channels, n_classes=3, image_dim=(512, 512)):
#     """
#     One hot
#     """

#     one_hot = np.zeros(im_dim[0], im_dim[1], n_classes)





def get_data(image_path, mask_path, min_hu=200, max_hu=700, n_classes=3):

    """
    Load the nifti data as arrays
    """

    im = nib.load(image_path)
    msk = nib.load(mask_path)

    im_data = im.get_data()
    msk_data = msk.get_data()


    def preprocessing(im1):
        im1 = np.moveaxis(im1, -1, 0)
        im1 = im1.astype(np.float32)
        return im1

    
    im_data = preprocessing(im_data)
    im_data = np.expand_dims(im_data, axis=-1)

    msk_data = preprocessing(msk_data)

    # Use tensorflow for a one-hot encoding
    msk_data_onehot = one_hot(msk_data, depth=4, dtype='float32')
    msk_data_onehot = msk_data_onehot.eval(session=tf.compat.v1.Session())    
    # msk_data = msk_data.numpy()

    im_data = (im_data - min_hu)/(max_hu - min_hu)



    return im_data, msk_data_onehot


# def adjustData(img,mask,flag_multi_class,num_class):
#     if(flag_multi_class):
#         img = img / 255
#         mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
#         new_mask = np.zeros(mask.shape + (num_class,))
#         for i in range(num_class):
#             #for one pixel in the image, find the class in mask and convert it into one-hot vector
#             #index = np.where(mask == i)
#             #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
#             #new_mask[index_mask] = 1
#             new_mask[mask == i,i] = 1
#         new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
#         mask = new_mask
#     elif(np.max(img) > 1):
#         img = img / 255
#         mask = mask /255
#         mask[mask > 0.5] = 1
#         mask[mask <= 0.5] = 0
#     return (img,mask)


def trainGeneratorArray(batch_size,image_array,mask_array,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1):
    '''
    Load from array

    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''

    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow(
        image_array,
        batch_size = batch_size,
        seed = seed)
    mask_generator = mask_datagen.flow(
        mask_array,
        batch_size = batch_size,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        # img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)


# def testGeneratorArray(test_path,num_image = 30,target_size = (256,256),flag_multi_class = False,as_gray = True):
#     """
#     Load from array

#     """
#     for i in range(num_image):
#         img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray)
#         img = img / 255
#         img = trans.resize(img,target_size)
#         img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
#         img = np.reshape(img,(1,)+img.shape)
#         yield img


def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)



def testGenerator(test_path,num_image = 30,target_size = (256,256),flag_multi_class = False,as_gray = True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray)
        img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img


def geneTrainNpy(image_path,mask_path,flag_multi_class = False,num_class = 2,image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True):
    image_name_arr = glob.glob(os.path.join(image_path,"%s*.png"%image_prefix))
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):
        img = io.imread(item,as_gray = image_as_gray)
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
        mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr


def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255



def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)


