# Business-Fraud-Detection-using-Machine-Learning-
Created a machine learning model for business fraud detection by ANN, Random Forest Classifier and XGBoost Classifier .
# Business Fraud Detection Using Machine Learning

This repository contains a comprehensive solution for detecting business fraud using machine learning techniques. The goal is to identify potentially fraudulent activities by analyzing various features of business transactions. This project includes data preprocessing, feature engineering, model training, and evaluation processes.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Data](#data)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Evaluation](#evaluation)


## Introduction

Business fraud detection is a critical task in ensuring the integrity and trustworthiness of financial transactions. By leveraging machine learning, we can build models that learn from historical data to identify patterns associated with fraudulent activities. This project aims to provide an end-to-end solution for detecting such fraud using various machine learning algorithms.

## Features

- Exploratory Data Analysis (EDA)
- Data cleaning including handling missing values, outliers, and multi-collinearity
- Feature engineering and selection
- Implementation of multiple machine learning algorithms
- Model evaluation and comparison
- Handling class imbalance
- Hyperparameter tuning
- Visualization of results

## Data

The dataset used in this project contains records of business transactions, including various features that describe the nature of each transaction. For confidentiality reasons, the dataset is not included in this repository. However, you can use any publicly available fraud detection dataset or your proprietary data. The required format and preprocessing steps are documented in the `data_preprocessing.ipynb` notebook.

## Installation

To get started, clone this repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/business-fraud-detection.git
cd business-fraud-detection
pip install -r requirements.txt
```

## Usage

1. **Data Preprocessing**: Prepare your data by running the `data_preprocessing.ipynb` notebook. This will clean the data, handle missing values, outliers, and multi-collinearity, and create necessary features.

2. **Model Training**: Train machine learning models by running the `model_training.ipynb` notebook. This notebook includes implementations of various algorithms such as Artificial Neural Networks, Random Forest, and XGBoost.

3. **Evaluation**: Evaluate the performance of the models using the `model_evaluation.ipynb` notebook. This notebook provides metrics like accuracy, precision, recall, F1-score, and AUC-ROC.

## Models

### Exploratory Data Analysis (EDA)

- **Data Cleaning**: Handle missing values, outliers, and multi-collinearity.
  - Checking for missing data
  - Checking for multi-collinearity
  - Checking for outliers
  - Selecting variables to be included in the model

- **Summary and Explanation**: 
  - `oldbalanceOrg` and `newbalanceOrg` are perfectly correlated because these columns represent the original and new balances in the sender's account after the transaction.
  - `oldbalanceDest` and `newbalanceDest` are perfectly correlated because these columns represent the original and new balances in the recipient's account.
  - `nameOrig` and `nameDest` are mass categorical variables that can be key factors in predicting fraudulent customers.

- **Key Variables**:
  - `step`
  - `type`
  - `amount`
  - `oldbalanceOrg`
  - `oldbalanceDest`
  - `isFraud`
  - `isFlaggedFraud`

### Data Preprocessing

- **Normalization**: Normalize `amount`, `oldbalanceOrg`, and `oldbalanceDest` to avoid dominance of significantly larger values.
- **Encoding**: Apply One Hot Encoding on the `type` feature.
- **Handling Class Imbalance**:
  - Why does class imbalance affect model performance?
    - Bias toward the majority class
    - Reduced sensitivity for the minority class
    - Low precision due to a high false positive rate
    - Difficulty learning patterns due to limited samples of the minority class
    - Skewed decision thresholds
  - Proposed Solution:
    - Combine oversampling and undersampling to balance the dataset
    - Use all fraudulent transactions and subsample non-fraudulent transactions to hit the target rate

### Model Implementations

- **Artificial Neural Network (ANN)**:
  - **Architecture**: Sequential model with an input layer, one hidden layer, and an output layer.
    - Input Layer: 64 neurons with ReLU activation.
    - Regularization: Dropout layers (30% dropout rate) after the input and hidden layers.
    - Hidden Layer: 32 neurons with ReLU activation.
    - Output Layer: Single neuron with sigmoid activation.
  - **Compilation**: Adam optimizer, binary cross-entropy loss.
  - **Training**: 100 epochs, batch size of 512.
  - **Evaluation Metrics**: True Positives, True Negatives, False Positives, False Negatives, Precision, Recall.

- **Random Forest Classifier**:
  - **Configuration**: 100 decision trees, OOB scoring disabled.
  - **Training**: Trained on the provided training data.
  - **Predictions**: Predictions on both training and test data.
  - **Evaluation Metrics**: True Positives, True Negatives, False Positives, False Negatives, Precision, Recall.

- **XGBoost Classifier**:
  - **Model**: XGBoost Classifier.
  - **Training**: Trained using the training data with AUC-PR as the evaluation metric.
  - **Predictions**: Predictions on both training and test data.
  - **Evaluation Metrics**: True Positives, True Negatives, False Positives, False Negatives, Precision, Recall.

## Evaluation

Model performance is evaluated using a variety of metrics to ensure robustness. The evaluation process includes:

- Confusion Matrix
- Precision, Recall, and F1-score
- AUC-ROC Curve
- Cross-validation

### Overall Performance Comparison

- **ANN Model**:
  - F1-score on the training set: 0.9500
  - F1-score on the test set: 0.9493

- **Random Forest**:
  - F1-score on the training set: 1.0 (perfect score)
  - F1-score on the test set: 0.9992

- **XGBoost**:
  - F1-score on the training set: 0.9967
  - F1-score on the test set: 0.9963

### Conclusion

The Random Forest model works best among the tested models.

