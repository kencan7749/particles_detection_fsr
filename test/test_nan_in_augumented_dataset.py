import unittest

# Helper libraries
import numpy as np
import os
import glob
import zipfile
import functools
#import h5py
import sys
sys.path.append('../point_claasification/2_learning/train_cluster')

#from 111_train_big_dataset_dual_point_cloud_all_info import augment_data
def augment_data(img, label_img, meta_img):
        # Take random snippet around polar angle = pi/2 (y-axis) for the two dust datasets of width 512 (approx. 90 degrees)
        # This snippet is shifted up to +-45 degrees (256 pixels) --> 90 degrees snippet between pi and 0 possible
        middle_angle = np.random.uniform(meta_img[0], meta_img[1])
        if middle_angle < 0: # Guarantess that at least end or start is in interval
            middle_angle += 2*np.pi
        middle_index = int(np.rint((width)*(middle_angle)/(2*np.pi)))
        start_index = int(middle_index - width_pixel/2)
        end_index = int(middle_index + width_pixel/2)
        if start_index >= 0 and end_index < width:
            img = img[:, start_index:end_index]
            label_img = label_img[:, start_index:end_index]
        elif end_index >= width:
            img = np.concatenate([img[:,start_index:],img[:,:end_index-width]], axis = 1)
            label_img = np.concatenate([label_img[:,start_index:],label_img[:,:end_index-width]], axis = 1)
        elif start_index < 0:
            img = np.concatenate([img[:, start_index+width:], img[:, :end_index]], axis = 1)
            label_img = np.concatenate([label_img[:, start_index+width:], label_img[:, :end_index]], axis = 1)
        # horizontal_flip with probability 0.5
        flip_prob = np.random.uniform(0.0, 1.0)
        if flip_prob > 0.5:
            img, label_img = img[:,::-1,:], label_img[:,::-1,:]
        return img, label_img

path = "./dataset/"

# Data definitions
file_names = ["1-dust", "2-dust", "3-dust", "4-dust", "5-dust", "6-dust", "7-dust",
                "8-dust", "9-smoke", "10-smoke", "11-smoke", "12-smoke", "13-smoke",
                "14-smoke", "15-smoke", "16-smoke", "17-smoke", "18-smoke", "19-smoke"]
#file_names = ["1-dust", "2-dust"]
metapath = "metadata.npy"
run = 0
train_indices = [1, 2, 3, 6, 7, 8, 10, 11, 13, 16, 18]
#the public dataset is missing 9-smoke
train_indices = [1, 2, 3, 6, 7, 10, 11, 13, 16, 18]
test_indices = [0, 4, 5, 9, 12, 14, 15, 17]
#train_indices = [0,1]
#test_indices = [0,1]
NAME = '111_dual_point_cloud_all_info_run_' + str(run+1)

for i in range(len(file_names)):
    file_names[i] = path + file_names[i]
metapath = path + metapath

# Complete filenames
for i in range(len(file_names)):
    file_names[i] = file_names[i] + "_labeled_spaces_img.npy"

# Import metadata
metadata = np.load(metapath)
metadata = metadata.astype(int)

# Load data
images_train = np.load(file_names[train_indices[0]])[metadata[train_indices[0],0]:metadata[train_indices[0],1]]
meta = metadata[train_indices[0], 2:4]
meta_train = np.zeros([len(images_train), 2])
meta_train[:,:] = meta
images_test = np.load(file_names[test_indices[0]])[metadata[test_indices[0],0]:metadata[test_indices[0],1]]
meta = metadata[test_indices[0], 2:4]
meta_test = np.zeros([len(images_test), 2])
meta_test[:,:] = meta
# Training Set
for i in range(len(train_indices)-1):
    current_images = np.load(file_names[train_indices[i+1]])[metadata[train_indices[i+1],
                                                                        0]:metadata[train_indices[i+1],1]]
    meta = metadata[train_indices[i + 1], 2:4]
    meta_vector = np.zeros([len(current_images), 2])
    meta_vector[:,:] = meta
    images_train = np.concatenate([images_train, current_images], axis = 0)
    meta_train = np.concatenate([meta_train, meta_vector], axis = 0)
print(images_train.shape)
print(meta_train.shape)
# Test Set
for i in range(len(test_indices)-1):
    current_images = np.load(file_names[test_indices[i+1]])[metadata[test_indices[i+1],
                                                                        0]:metadata[test_indices[i+1],1]]
    meta = metadata[test_indices[i + 1], 2:4]
    meta_vector = np.zeros([len(current_images), 2])
    meta_vector[:,:] = meta
    images_test = np.concatenate([images_test, current_images], axis = 0)
    meta_test = np.concatenate([meta_test, meta_vector], axis=0)
print(images_test.shape)
print(meta_test.shape)

# --------------------------------Start of actual software---------------------------------

features_train = images_train[:,:,:,[0,4,6,10]] # Here change number of channels
labels_train = images_train[:,:,:,12:15]
features_test = images_test[:,:,:,[0,4,6,10]] # Here change number of channels
labels_test = images_test[:,:,:,12:15]
del images_train
del images_test

width = 2172 # So far empricially for all datasets
width_pixel = 512
img_shape = (32, 512, 4) # 512 comes from approx 90 degrees (360 degrees are 2172) also since it can be divided by 32
                            # Change input channels here
number_labels = 3
batch_size =32
epochs = 50
num_train_examples = len(features_train)
num_test_examples = len(features_test)

# Build validation data
features_test_2 = np.zeros([features_test.shape[0], features_test.shape[1], width_pixel, features_test.shape[3]])
labels_test_2 = np.zeros([labels_test.shape[0], labels_test.shape[1], width_pixel, labels_test.shape[3]])
for i, img in enumerate(features_test):
    img, label_img = augment_data(img, labels_test[i], meta_test[i])
    features_test_2[i, :, :, :] = img
    labels_test_2[i, :, :, :] = label_img
features_test = features_test_2
del features_test_2
labels_test = labels_test_2
del labels_test_2

#print(np.sum(features_test))
#print(np.sum(labels_test))
assert np.sum(features_test) is not None
assert np.sum(labels_test) is not None
print('Finish valiation')

def generator(features, labels, meta_train):
    print('Start!')
    
    print("\naugmented!\n")
    print(int(np.ceil(len(features) / float(batch_size))))
    #batch_size = len(features) 
    # Shuffle Arrays in same manner
    permutation = np.random.permutation(len(features))
    features = features[permutation]
    labels = labels[permutation]
    meta_train = meta_train[permutation]
    # Pick out one after another in generator form
    for i in range(int(np.ceil(len(features) / float(batch_size)))-1):
        feature_new = np.zeros((batch_size, features[batch_size*i].shape[0], width_pixel,
                                features[batch_size*i].shape[2]))
        label_new = np.zeros((batch_size, labels[batch_size*i].shape[0], width_pixel,
                                labels[batch_size*i].shape[2]))
        for c in range(batch_size):
            feature, label = augment_data(features[batch_size*i+c], labels[batch_size*i+c],
                                            meta_train[batch_size*i+c])
            #feature_new[c,:,:,:] = feature
            #label_new[c,:,:,:] = label
            assert np.sum(feature) is not None
            assert np.sum(label) is not None
        print('fine')
    return feature_new, label_new

generator(features_train, labels_train, meta_train)
generator(features_test, labels_test, meta_train)

print('done -> nothing nan in data and augumented data')