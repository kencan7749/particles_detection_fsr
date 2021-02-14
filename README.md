The keras model to classify airborne particles in LiDAR data.

[Paper]

Stanislas et al., 2019 [Airborne Particle Classification in LiDAR Point Clouds Using Deep Learning](https://eprints.qut.edu.au/133596/)

[History] 

2021/02/13 KS created v0.5

# How to use 

This provides four steps to perform to train and evaluate this model.

1. Preprocessing
2. Training 
3. Evaluation
4. Prediction

### Environment

I confirmed to run these scripts in this environment

```python
Python3.6.10
tqdm
tensorflor-gpu= 1.15.0
numpy
scipy
scikit-learn = ??
```



## 1. Preprocessing

[Code]

```python
point_classification/1_preprocessing/transform_pointcloud_to_image_representation.py
```

Preprocessing 3D LiDAR scan into 2D LiDAR images with each pixel corresponding to an emitted ray and the rows and columns of this image correspond to the vertical and horizontal angular resolution of the LiDAR sensor respectively.

##### Requirements 

The preprocessing data should be stored at `point_classification/data`. Note that the original [data repository](https://cloudstor.aarnet.edu.au/plus/s/oQwj9AkaLlqNU1a) was not stored `9-dust` dataset. I skipped this dataset in this analysis.

[Input]

The numpy files, whose name starts like "1-dust", "2-dust", ..."14-smoke". These file is specified at line 21,  `file_names` variables.

[Output]

Preprocessed files, whose name ends like "1-dust~img.npy" "2-dust~img.npy", "14-smoke~img.npy".

## 2.Training

[Code] (for example 112 model)

```python
point_classification/2_learning/train_cluster/112_train_big_dataset_dual_point_cloud_all_info_validated_update.py
```

Train the U-Net architecture model to perform particle detection model from preprocessed images.

The 9 training files are provided. In my guess, the file names "dual" means "Multi-Echo" in original paper, while "single" means no "Multi-Echo". "all_info" means using both features "Geometry" and "Intensity". 

##### Requirements

The preprocessed data (ex. 1-dust~img.npy) should be stored at `point_classification/data`.

The `metadata.npy` should also be stored at `point_classification/data`.

[Input]

- The preprocessed data (ex. 1-dust~img.npy).  These file is specified at line 37,  `file_names` variables. The line 45 and 47, determined whether these files are used for training or validation.
- The`metadata.npy`

[Output]

- Log files at directory `point_classification`. (it should be updated to locate at`point_classification/logs`)
- Weight files at directory at `point_classification/models`.

## 3. Evaluation (under review)

[Code]

```python
point_classification/2_learninig/evaluation/evaluate_performance_on_model.py
```

Evaluate the performance of the trained model by precision, recall and f1 score in scikit-learn.

##### Requirements 

Trained model files should be located at `point_classification/models'` . These file is specified at line 45, `weight_names` variable to select which trained models are used for evaluation. The preprocessed data (ex. 1-dust~img.npy) should also be needed.  These file is specified at line 30,  `file_names` variables, and line 33 `test_indices` determines which data were used for evaluation.

[Input]

- The trained model.
- The preprocessed image data

[Output]

eval_result.txt located at`point_classification/evals`. This file contains the metric results of precision, recall and f1-score for all iteration model specified at `weight_names` .



###### Note that currently the evaluation script is less computational efficiency and takes much time to finish. Also I am reviewing scikit-learn version since I observed precision and recall both get 0.0 with warning. 

## 4.Prediction (under review)

[code]

For generating image

```python
point_classification/2_learning/prediction/predict_input_to_img.py
```

For generating point cloud 

```python
point_classification/2_learning/prediction/predict_input_to_pcl.py
```

###### requirement

```python
numpy
tensorflow-gpu=1.15.0
scipy
```

