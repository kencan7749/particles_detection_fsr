import os.path as osp
import argparse
import numpy as np
from tqdm import tqdm
#import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score
from sklearn.metrics import average_precision_score
from return_model import return_model
from generator import generator
import tensorflow as tf
import tensorflow.contrib as tfcontrib
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import backend as K
from tensorflow.python.client import session as sess

session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1
)

tf.set_random_seed(7)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

parser = argparse.ArgumentParser(description='set model name and run')
parser.add_argument("model_name", help="set model name to use (like '112', '212'")
parser.add_argument("run_number", help="set run number (like 1, 2, ... 10")

args = parser.parse_args()

file_names = ["/1-dust", "/2-dust", "/3-dust", "/4-dust", "/5-dust", "/6-dust", "/7-dust", "/8-dust", "/9-smoke",
              "/10-smoke", "/11-smoke", "/12-smoke", "/13-smoke", "/14-smoke", "/15-smoke", "/16-smoke",
              "/17-smoke", "/18-smoke", "/19-smoke"]
test_indices = [0, 4, 5, 9, 12, 14, 15, 17]
test_indices = [0]
file_path = "./dataset/"

weight_names_dict = {"111": "weights_big_dataset_111_dual_point_cloud_all_info_run_", 
                    "112" : "weights_big_dataset_112_dual_point_cloud_all_info_v_u_run_",
                    "121" : "weights_big_dataset_121_dual_point_cloud_geometry_run_", 
                    "122" : "weights_big_dataset_122_dual_point_cloud_geometry_v_u_run_",
                    "131" : "weights_big_dataset_131_dual_point_cloud_intensities_run_",
                    "132" : "weights_big_dataset_132_dual_point_cloud_intensities_v_u_run_",
                    "211" : "weights_big_dataset_211_single_point_cloud_all_info_run_", 
                    "212" : "weights_big_dataset_212_single_point_cloud_all_info_v_u_run_",
                    "221" : "weights_big_dataset_221_single_point_cloud_geometry_run_", 
                    "222" : "weights_big_dataset_222_single_point_cloud_geometry_v_u_run_",
                    "231" : "weights_big_dataset_231_single_point_cloud_intensities_run_", 
                    "232" : "weights_big_dataset_232_single_point_cloud_intensities_v_u_run_"}
                
model_name = args.model_name
weight_name = weight_names_dict[model_name]
run = args.run_number
#weight_names = weight_names[::-1] # Flip weight_names
weights_path = "./models/"
#weights_path = "./dataset/trained_models/"

eval_dir = "./evals/"

width_pixel = 2172
width_pixel = width_pixel - width_pixel % 16
batch_size = 1

eval_results= '-'.join([model_name, run, 'eval_results.txt'])
with open(osp.join(eval_dir, eval_results), 'w') as f:
    f.write('All results for all models for all runs for the following datasets:\n')
    for ind in test_indices:
        f.write('%s\n' % file_names[ind])



weights_file = weights_path + weight_name + str(run) + ".hdf5"
print("Currently investigating: " + weights_file)
# For all of the scans load feature and label vector
for i, index in enumerate(test_indices):
    file_name = file_names[index]
    print(file_name)
    pc_file = file_path + file_name + "_labeled_spaces_img.npy"
    image_test = np.load(pc_file)[:,:,:width_pixel,:]
    if "dual" in weights_file:
        features = image_test[:, :, :, [0, 4, 6, 10]]
        if "geometry" in weights_file:
            features = image_test[:, :, :, [0, 6]]
        elif "intensities" in weights_file:
            features = image_test[:, :, :, [4, 10]]
    else:
        features = image_test[:, :, :, [0, 4]]
        if "geometry" in weights_file:
            features = image_test[:, :, :, [0]]
        elif "intensities" in weights_file:
            features = image_test[:, :, :, [4]]
    labels_target_temp = image_test[:, :, :, len(image_test[0, 0, 0, :]) - 3:]
    # Load Model for each record
    model = return_model(features.shape[1:4], weights_file)
    # Do inference
    labels_pred_temp = model.predict_generator(generator(features, labels_target_temp, batch_size, width_pixel),
                                                steps=int(np.floor(len(features) / float(batch_size))))
    del features
    # Filter out zero values returned by LiDAR --> Nice effect: directly transformed to vector representation
    labels_target_temp = labels_target_temp[
        ((image_test[:, :, :, 0] != 0) | (image_test[:, :, :, 1] != 0)) | (image_test[:, :, :, 2] != 0)]
    labels_pred_temp = labels_pred_temp[
        ((image_test[:, :, :, 0] != 0) | (image_test[:, :, :, 1] != 0)) | (image_test[:, :, :, 2] != 0)]
    del image_test
    # Stack them onto each other to do evaluation afterwards
    if i == 0:
        labels_target = labels_target_temp
        labels_pred = labels_pred_temp
    else:
        labels_target = np.concatenate((labels_target, labels_target_temp), axis = 0)
        labels_pred = np.concatenate((labels_pred, labels_pred_temp), axis = 0)

    del labels_target_temp, labels_pred_temp
print(labels_target.shape)
print(labels_pred.shape)
# Do the evaluation for this run (and all records together)

# Allocate Vector which will contain all labels
y_pred_particle = np.zeros((labels_pred.shape[0],1))
y_pred_dust = np.zeros((labels_pred.shape[0],1))
y_pred_fog = np.zeros((labels_pred.shape[0], 1))
y_target_particle = np.zeros((labels_target.shape[0],1))
y_target_dust = np.zeros((labels_target.shape[0], 1))
y_target_fog = np.zeros((labels_target.shape[0], 1))

# Write in prediction vector
# Write in target vector
for i, label in tqdm(enumerate(labels_pred)):
    if np.argmax(label) == 1 or np.argmax(label) == 2:
        y_pred_particle[i,0] = 1
    if np.argmax(label) == 1:
        y_pred_dust[i,0] = 1
    elif np.argmax(label) == 2:
        y_pred_fog[i,0] = 1

    label_target = labels_target[i]
    if np.argmax(label_target) == 1 or np.argmax(label_target) == 2:
        y_target_particle[i,0] = 1
    if np.argmax(label_target) == 1:
        y_target_dust[i,0] = 1
    elif np.argmax(label_target) == 2:
        y_target_fog[i,0] = 1
del labels_pred
del labels_target

with open(osp.join(eval_dir, eval_results), 'a') as f:
    f.write('Evaluation parameters:\n')
    f.write('Model: ' + weights_file + '\n')

    #f.write('No bias: %r\n' % no_bias)
    #f.write('Cropped data: %r\n' % cropped)
    f.write('\n\nEvaluation results:\n')

    # Compute performance scores
    print('\n')
    # Particle ---------------------------------------------
    precision = precision_score(y_target_particle, y_pred_particle, pos_label=1)
    print('Precision for predicting Particle: %f' % precision)
    f.write('Precision for predicting Particle: %f\n' % precision)

    recall = recall_score(y_target_particle, y_pred_particle, pos_label=1)
    print('Recall for predicting Particle: %f' % recall)
    f.write('Recall for predicting Particle: %f\n' % recall)

    f1 = f1_score(y_target_particle, y_pred_particle, pos_label=1)
    print('F1 score for predicting Particle: %f\n' % f1)
    f.write('f1-score for predicting Particle: %f\n' % f1)

    # Dust ---------------------------------------------
    precision = precision_score(y_target_dust, y_pred_dust, pos_label=1)
    print('Precision for predicting Dust: %f' % precision)
    f.write('\nPrecision for predicting Dust: %f\n' % precision)

    recall = recall_score(y_target_dust, y_pred_dust, pos_label=1)
    print('Recall for predicting Dust: %f' % recall)
    f.write('Recall for predicting Dust: %f\n' % recall)

    f1 = f1_score(y_target_dust, y_pred_dust, pos_label=1)
    print('F1 score for predicting Dust: %f\n' % f1)
    f.write('f1-score for predicting Dust: %f\n' % f1)

    # Fog ---------------------------------------------
    precision = precision_score(y_target_fog, y_pred_fog, pos_label=1)
    print('Precision for predicting Fog: %f' % precision)
    f.write('\nPrecision for predicting Fog: %f\n' % precision)

    recall = recall_score(y_target_fog, y_pred_fog, pos_label=1)
    print('Recall for predicting Fog: %f' % recall)
    f.write('Recall for predicting Fog: %f\n' % recall)

    f1 = f1_score(y_target_fog, y_pred_fog, pos_label=1)
    print('F1 score for predicting Fog: %f\n' % f1)
    f.write('f1-score for predicting Fog: %f\n' % f1)

    # Can only use this if both classes are at least predicted once
    print('Classification Report')
    f.write('Classification Report\n')
    if len(np.unique(y_pred_particle)) > 1:
        cr = classification_report(y_target_particle, y_pred_particle)
        print(cr)
        f.write(cr)
    f.write('\n')