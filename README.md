# CapstoneProject

## Group Members
- Wambui Kamotho
- Ian Bett
- Sharon Paul
- Frankline Kipchumba
- Habshi Pedro

---

## Project Summary

This project uses camera trap image data to build a machine learning model that can identify wildlife species. The model will support conservation efforts by enabling automated monitoring of species diversity and activity.

---

## 1. Business Problem

Conservationists have recently transitioned from having photographers lie in the bushes or up in trees taking photos to camera traps. These are cameras set up in the wild that take photos of animals using motion sensors or heat-detecting technology. These camera traps generate large amounts of data, more than conservationists can reasonably sort through and analyze. We are creating an image classifier that can identify the type of animal in an image using neural networks.


---

## Objectives

Create a model that will classify images of wildlife by their species, enabling conservationists to understand animals’ behavior patterns, movement, and population trends.

### Main Objective

- Learn more about different animals and their interactions with each other.
- Support the tourism industry by allowing guides to know the exact locations of different animals with minimal disruption of habitats, and optimizing resource use.
- Get a fuller understanding of animal patterns and spot deviations quickly..
Biodiversity conservation by quickly identifying struggling species or invasive species.


---

## Stakeholders

- KWS(Kenya wildlife service)
- Safari companies
- Academics
- Private conservation groups


## 2. Data Understanding

This data is from an ongoing competition on DrivenData. The dataset was compiled by the Wild Chimpanzee Foundation and the Max Planck Institute for Evolutionary Anthropology. The camera traps are located in Taï National Park in Ivory Coast.

The competition can be accessed through the following link: https://www.drivendata.org/competitions/87/competition-image-classification-wildlife-conservation/page/483/

Each image in this data set is labeled as one of 7 creatures: birds, civets, duikers, hogs, leopards, monkeys, rodents. Images with no animal on them are labeled blank.


### Source of Data
- Folders containing camera trap images.
- CSV files containing image metadata and species labels.

### Validity
- The dataset is collected from automated camera traps and labeled manually by researchers.

### Features and What They Represent
- `id`: Unique identifier for the image.
- `filepath`: Path to the image file.
- `site`: Camera trap location site ID.
- Label Columns (e.g., `leopard`, `bird`, `hog`, etc.): Binary indicators for species presence.

### Metric of Success
- Highest **Micro and Macro ROC-AUC** for multi-label classification.



## 3. DATA CLEANING

## Data Merging Summary

We loaded:
- `train_features.csv`: includes image ID and relative file path.
- `train_labels.csv`: includes binary indicators for each species per image.

We merged both DataFrames on the `id` column and created a new column `filepath` with the full path to each image, so Keras can easily locate them.


##  4. EXPLORATORY DATA ANALYSIS (EDA)
### 4.1 - Univariate Analysis

In this section, we explore the distribution of individual variables, especially the occurrence of each species label. This helps us understand class balance and the overall biodiversity captured by the camera traps.

## 4.2 - Bivariate Analysis
In this section, we explore how species occurrences vary by site. This helps us identify spatial patterns in wildlife presence and habitat preferences.

## 4.3 - Multivariate Analysis

In this section, we explore inter-species co-occurrence patterns by computing correlation across label columns. This can reveal ecological relationships or tagging tendencies.

# 5. FEATURE ENGINEERING
Feature engineering is crucial in machine learning, especially when working with image data. Here, we'll focus on the following steps:

1. Image Preprocessing: Transforming raw image data into a format suitable for training.

2. Data Augmentation: Expanding the dataset artificially by applying transformations.

3. Normalization: Scaling pixel values to a range that is optimal for training.

4. Data Splitting: Dividing the dataset into training, validation, and test sets.

## 5.1 Image Preprocessing
To ensure all images are of the same size, we resize them to 224x224 pixels, a common input size for image classification models.

## 5.2 Data Augmentation
Data augmentation helps to artificially expand the dataset by introducing slight variations of the images (such as rotation, flipping, and zooming), which helps the model generalize better to unseen data. We’ll apply augmentation only to the training set.

## 5.3 Normalization
Normalization is the process of scaling the pixel values to a range that is more suitable for the neural network to train. We'll normalize the pixel values to the range [0, 1] by dividing by 255.

## 5.4 Data Splitting
To evaluate model performance effectively, we split the data into three sets:

Training Set: 80% of the data, used for model training.

Validation Set: 10% of the data, used to tune the model’s hyperparameters.

Test Set: 10% of the data, used to evaluate the final model’s performance.

## 6. MODELLING

In this section, we build and evaluate several deep learning models for multi-label image classification. Our goal is to identify the best-performing model for detecting species in camera trap images.

We begin with a **baseline model** to establish a benchmark, followed by three advanced models.
-  Model 1 - MobileNetV2.
-  Model 2 - ResNet50.
-  Model 3 - Custom CNN with Regularization.

## 7. Model Evaluation
### Evaluation Metrics

For this multi-label image classification task, we use **ROC-AUC (Micro and Macro averaged)** as our primary metrics of success.

- **Micro ROC-AUC** evaluates overall model performance by aggregating all true positives, false positives, and false negatives, making it sensitive to class imbalance.
- **Macro ROC-AUC** calculates the AUC for each class and averages them, giving equal weight to all species regardless of frequency.

These metrics are chosen over simple validation accuracy because:
- Validation accuracy is overly strict in multi-label problems, requiring an exact match of all labels.
- ROC-AUC provides a threshold-independent measure of the model's ability to distinguish between classes.
- Micro ROC-AUC reflects overall performance, while Macro ROC-AUC ensures minority classes are also considered.

Therefore, **the model with the highest combined Micro and Macro ROC-AUC is selected as the best-performing model**.

### Results Summary

| Model         | Micro ROC-AUC | Macro ROC-AUC | Exact Match |
|---------------|----------------|----------------|--------------|
| Baseline CNN  | 0.5208         | 0.5009         | 0.1101       |
| MobileNetV2   | 0.5146         | 0.4994         | 0.0980       |
| ResNet50      | 0.5245         | 0.4914         | 0.0146       |
| Custom CNN    | **0.5260**     | **0.4996**     | 0.0055       |

Based on the highest **Micro and Macro ROC-AUC**, the **Custom CNN** is selected as the best-performing model.


## 8. CONCLUSIONS AND RECOMMENDATIONS
### 8.1 Conclusion
This project is a multi-label image classification pipeline for camera trap images. Four CNN-based models were trained and evaluated: a Baseline CNN, MobileNetV2, ResNet50, and a Custom CNN with regularization.

**Conclusions:**
- The baseline model achieved better Micro and Macro ROC-AUC  compared to MobileNetV2 and ResNet50.
- ResNet50, benefited from complex feature extraction but may require fine-tuning for optimal performance.
- The Custom CNN gave more hope with incorporation with Batch Normalization and Dropout to mitigate overfitting.

### 8.2 Recommendations

The recommendations:

- **Class Imbalance:** Apply techniques such as class weighting, oversampling rare classes, or using focal loss to improve learning for minority species.
- **Label Quality Review:** Perform manual checks or automated noise reduction techniques to improve the quality of image labels.
- **Increase Training Duration:** Train models for more epochs with early stopping to allow better convergence.
- **Use Higher Resolution Images:**

### 8.3 Next Steps

To further improve performance and build on this foundation:

- **Fine-tune Pretrained Models:**
- **Hyperparameter Tuning:** Optimize learning rates and regularization parameters using grid search or randomized search.
- **Data Augmentation:** Apply stronger augmentations and variations.
- **Test-Time Augmentation:** Improve strength on the prediction by predicitng over multiple augmented versions of the same image.
- **Detect animals in images using other models like R-CNN:**
-  

