# Credit Card Fraud Detection
## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Objective](#objective)
- [Technologies Used](#technologies-Used)
- [Project Workflow](#project-Workflow)
- [Results](#results)
- [How to Run](#how-to-run)
- [Future Work](#future-Work)
- [Contributors](#contributors)
---
## Introduction
Credit card fraud is a major concern for both financial institutions and customers. This project aims to develop a machine learning-based solution to detect fraudulent credit card transactions efficiently and accurately. The system leverages advanced algorithms to differentiate between legitimate and fraudulent transactions.
---

## Dataset
The dataset used in this project is the [Kaggle Credit Card Fraud Detection Dataset.](#https://www.kaggle.com/datasets/kartik2112/fraud-detection?select=fraudTrain.csv)
This is a simulated credit card transaction dataset containing legitimate and fraudulent transactions from the duration of 1st Jan 2019 to 31st Dec 2020. It covers credit cards of 1000 customers doing transactions with a pool of 800 merchants.

- Features : 23

## Objective
The main objective of this project is to build a model that can:

1. Accurately classify transactions as fraudulent or legitimate.


2. Minimize false negatives (Recall).


3. Handle the imbalance in the dataset effectively.

## Technologies Used
- ### Programming Language: Python

- ### Libraries:

   - Data Analysis: pandas, numpy

   - Data Visualization: matplotlib, seaborn

   - Machine Learning: scikit-learn

   - Model Evaluation: sklearn.metrics

## Project Workflow
1. ### Data Preprocessing:
   - column [trans_date_trans_time] that contains data and time of the process, we can split this column into year, month, day, hour, minute
   - calculate the age from column ['dob'] containing Date of Birth of Credit Card Holder
   - Handle missing.
   - handling categorical data
   - Normalize the data
   - Address data imbalance using techniques (Class Weights).
2. ### Exploratory Data Analysis (EDA):
   - Visualize the class distribution.
   - Identify correlations and trends in the data.
3. ### Feature Selection:
   - Use statistical tests to select relevant features.
4.  ### Model Selection:
    - Train binary clasification machine learning model (Logistic Regression, Random Forest).
5. ### Evaluation:
   - Use metrics Precision, Recall, F1-Score to evaluate model performance.
6. ### Deployment (optional):
   - Save the trained model using joblib or pickle for future predictions.


## Results
- ### Best Model: Random Forest

  - Accuracy: 99.7%

  - Precision: 99%

  - Recall: 84%

  - F1-Score: 90%

## How to Run
1. Creation of virtual environments
```Bash
python -m venv <name of your environment>
```
2. activation of environment
```Bash
<name of your environment>\Scripts\activate
```
3. Change the directory inside to the environment
```Bash 
cd <name of your environment>
```
4. creat folder in this directory
```Bash 
md src
```
5. Change the directory inside to src
```Bash
cd src
```
6. Colne this repository:
```Bash
git clone <url of repo >
```
7. install the required dependencies:
```Bash 
pip install -r requirements.txt
```
8. Run the credit_card_fraud_detection.py script:
```Bash 
python main.py <path of dataset>
```
## Future Work
- integrate the model into a real-time transaction monitoring system.
- Test the model on a larger, more diverse dataset.

## Contributors
- my Name: [lamiaakhairyibrahim](#https://github.com/lamiaakhairyibrahim)




