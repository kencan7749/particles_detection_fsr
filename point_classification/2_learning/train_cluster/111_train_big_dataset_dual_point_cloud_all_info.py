# Helper libraries
import numpy as np
import os
import sys
import glob
import zipfile
import functools
import h5py
import random as rn
import argparse
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(7)
rn.seed(7)

# TensorFlow and tf.keras
import tensorflow as tf
import tensorflow.contrib as tfcontrib
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import backend as K
from tensorflow.python.client import session as sess
from tensorflow.python.client import device_lib
#from tfdeterminism import patch
###for reproducibility
#patch() 

#def get_available_gpus():
#    local_device_protos = device_lib.list_local_devbices()
#    return [x.name for x in local_device_protos if x.device_type == 'GPU']

#os.environ["CUDA_VISIBLE_DEVICES"]="1,3"


session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1
)

tf.set_random_seed(7)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
 
parser = argparse.ArgumentParser()
parser.add_argument("--cluster", help="Runs script on cluster")
args = parser.parse_args()
gpu_count = 2
path = "./dataset/"
for run in range(10):

    # Data definitions
    file_names = ["1-dust", "2-dust", "3-dust", "4-dust", "5-dust", "6-dust", "7-dust",
                  "8-dust", "9-smoke", "10-smoke", "11-smoke", "12-smoke", "13-smoke",
                  "14-smoke", "15-smoke", "16-smoke", "17-smoke", "18-smoke", "19-smoke"]
    file_names = ["1-dust", "2-dust"]
    metapath = "metadata.npy"

    train_indices = [1, 2, 3, 6, 7, 8, 10, 11, 13, 16, 18]
    #the public dataset is missing 9-smoke
    train_indices = [1, 2, 3, 6, 7, 10, 11, 13, 16, 18]
    test_indices = [0, 4, 5, 9, 12, 14, 15, 17]
    #file_names = ["1-dust", "2-dust"]
    train_indices = [0,1]
    test_indices = [0,1]
    NAME = '111_dual_point_cloud_all_info_run_' + str(run+1)

    # In case we run it on the local pc
    if not args.cluster:
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
    batch_size = 32
    epochs = 50
    num_train_examples = len(features_train)
    num_test_examples = len(features_test)

    # Building the model

    def conv_block(input_tensor, num_filters):
        encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
        encoder = layers.BatchNormalization()(encoder)
        encoder = layers.Activation('relu')(encoder)
        encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
        encoder = layers.BatchNormalization()(encoder)
        encoder = layers.Activation('relu')(encoder)
        return encoder


    def encoder_block(input_tensor, num_filters):
        encoder = conv_block(input_tensor, num_filters)
        encoder_pool = layers.MaxPooling2D((1, 2), strides=(1, 2))(encoder)

        return encoder_pool, encoder


    def decoder_block(input_tensor, concat_tensor, num_filters):
        decoder = layers.Conv2DTranspose(num_filters, (1, 2), strides=(1, 2), padding='same')(input_tensor)
        decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        return decoder

    inputs = layers.Input(shape=img_shape)
    # 2144
    encoder0_pool, encoder0 = encoder_block(inputs, 16)
    # 1072
    encoder1_pool, encoder1 = encoder_block(encoder0_pool, 32)
    # 536
    encoder2_pool, encoder2 = encoder_block(encoder1_pool, 64)
    # 268
    encoder3_pool, encoder3 = encoder_block(encoder2_pool, 128)
    # 134
    center = conv_block(encoder3_pool, 256)
    # center
    decoder3 = decoder_block(center, encoder3, 128)
    # 268
    decoder2 = decoder_block(decoder3, encoder2, 64)
    # 536
    decoder1 = decoder_block(decoder2, encoder1, 32)
    # 1072
    decoder0 = decoder_block(decoder1, encoder0, 16)
    # 2144
    outputs = layers.Conv2D(number_labels, (1, 1), activation='softmax')(decoder0)

    model = models.Model(inputs=[inputs], outputs=[outputs])

    def dice_coeff(y_true, y_pred):
        smooth = 1.
        # Flatten
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
        #assert np.isnan(score.eval(session=tf.compat.v1.Session()))
        return score

    def dice_loss(y_true, y_pred):
        loss = 1 - dice_coeff(y_true, y_pred)
        return loss

    def bce_dice_loss(y_true, y_pred):
        loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
        #assert np.isnan(loss.eval(session=tf.compat.v1.Session()))
        tf.print(loss, output_stream=sys.stderr)
        return loss

    try:
        model = tf.keras.utils.multi_gpu_model(model, gpus=None) # add
        print('Multiple GPU!!')
    except:
        print('Single GPU...')
        pass
    model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[dice_loss, 'accuracy'])

    model.summary()

    save_model_weights = 'models/weights_big_dataset_' + NAME +'.hdf5'
    #save_model = 'models/model_big_dataset_always_update.hdf5'
    log_dir = 'logs\\' + NAME
    os.makedirs('models', exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    cp = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_weights, monitor='val_dice_loss', save_best_only=False, verbose=1)
    cp2 = tf.keras.callbacks.TensorBoard(log_dir='logs\\' + NAME, histogram_freq=0,
                              write_graph=False, write_images=False)
    import datetime
    class CheckNaN(tf.keras.callbacks.Callback):
        # コンストラクタ
        def __init__(self):
            self.last_acc, self.last_loss, self.last_val_acc, self.last_val_loss = None, None, None, None
            self.now_batch, self.now_epoch = None, None

            self.epochs, self.samples, self.batch_size = None, None, None

        # カスタム進捗表示 (表示部本体)
        def print_progress(self):
            epoch = self.now_epoch
            batch = self.now_batch

            epochs = self.epochs
            samples = self.samples
            batch_size = self.batch_size
            sample = batch_size*(batch)
            if np.isnan(self.last_loss) or np.isnan(self.dice_loss):
                
                print("\rRaise NaN Epoch %d/%d (%d/%d) -- acc: %f loss: %f dice_loss: %f" % (epoch+1, epochs, sample, samples, self.last_acc, self.last_loss,self.dice_loss), end='')
                assert 1 == 0

            # '\r' と end='' を使って改行しないようにする
            if self.last_val_acc and self.last_val_loss:
                # val_acc/val_loss が表示可能
                print("\rManual Epoch %d/%d (%d/%d) -- acc: %f loss: %f dice_loss: %f - val_acc: %f val_loss: %f" % (epoch+1, epochs, sample, samples, self.last_acc, self.last_loss, self.dice_loss,self.last_val_acc, self.last_val_loss), end='')
            else:
                # val_acc/val_loss が表示不可
                print("\rManual Epoch %d/%d (%d/%d) -- acc: %f loss: %f dice_loss: %f" % (epoch+1, epochs, sample, samples, self.last_acc, self.last_loss,self.dice_loss), end='')

        # fit開始時
        def on_train_begin(self, logs={}):
            print('\n##### Train Start ##### ' + str(datetime.datetime.now()))

            # パラメータの取得
            self.epochs = self.params['epochs']
            self.samples = self.params['samples']
            self.batch_size = self.params['batch_size']
            if self.batch_size is None:
                self.batch_size = batch_size

            print('epoch' + str(self.epochs))
            print('sampl' + str(self.samples))
            print('batch' + str(self.batch_size))

            #assert 1 == 0

            # 標準の進捗表示をしないようにする
            self.params['verbose'] = 0
        # batch開始時
        def on_batch_begin(self, batch, logs={}):
            self.now_batch = batch
            #assert self.batch
            print(batch)

        # batch完了時 (進捗表示)
        def on_batch_end(self, batch, logs={}):
            # 最新情報の更新
            self.last_acc = logs.get('acc') if logs.get('acc') else 0.0
            self.last_loss = logs.get('loss') if logs.get('loss') else 0.0
            self.dice_loss = logs.get('dice_loss') if logs.get('dice_loss') else 0.0
            #assert 1 == 0
            # 進捗表示
            self.print_progress()
        # epoch開始時
        def on_epoch_begin(self, epoch, log={}):
            self.now_epoch = epoch

        # epoch完了時 (進捗表示)
        def on_epoch_end(self, epoch, logs={}):
            # 最新情報の更新
            self.last_val_acc = logs.get('val_acc') if logs.get('val_acc') else 0.0
            self.last_val_loss = logs.get('val_loss') if logs.get('val_loss') else 0.0

            # 進捗表示
            #self.print_progress()
    cp3 = CheckNaN()
    # Function to augment data and keep randomly selected image of width "width_pixel"
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

    # Define Generator to save memory
    def generator(features, labels, meta_train):
        while True:
            print("\naugmented!\n")
            print(int(np.ceil(len(features) / float(batch_size))))
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
                    feature_new[c,:,:,:] = feature
                    label_new[c,:,:,:] = label
                assert not np.any(np.isnan(feature_new))
                assert not np.any(np.isnan(label_new))
                print(feature_new.shape, label_new.shape)
                yield feature_new, label_new

    #with tf.device('/gpu:0'):
    with tf.device('cpu:0'):
        print("tf.keras code in this scope will run on CPU")
    
        history = model.fit_generator(generator(features_train, labels_train, meta_train),
                                    steps_per_epoch=int(np.ceil(num_train_examples / float(batch_size))),
                                    epochs = epochs,
                                    validation_data=(features_test, labels_test),
                                    validation_steps=int(np.ceil(num_test_examples / float(batch_size))),
                                    callbacks=[cp, cp2, cp3])

    #history = model.fit(dataset,
    #                   steps_per_epoch=int(np.ceil(num_train_examples / float(batch_size))),
    #                   epochs=epochs,
    #                   validation_data=val_ds,
    #                   validation_steps=int(np.ceil(num_test_examples / float(batch_size))),
    #                   callbacks=[cp])

    #model.save(save_model)