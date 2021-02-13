[History] 

2021/02/13 KS created v0.5

# How to use 

1. Preprocessing

2. Training 

3. Evaluation

4. Prediction

   

## 1. Preprocessing

[code]

```python
point_classification/1_preprocessing/transform_pointcloud_to_image_representation.py
```

Preprocessing 

##### requirements 

```python
numpy
```

[input]



[output]





## 2.Training

[code] (for example 112 model)

```python
point_classification/2_learning/train_cluster/112_train_big_dataset_dual_point_cloud_all_info_validated_update.py
```



##### requirements

```.0python
tensorflow-gpu == 1.15.0
numpy
```

[input]



[output]



## 3. Evaluation

[code]

```python
point_classification/2_learninig/evaluation/evaluate_performance_on_model.py
```



##### requirements

```python
numpy
tqdm
tensorflow-gpu = 1.15.0
scikit-learn
```



[input]



[output]



## 4.Prediction

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

