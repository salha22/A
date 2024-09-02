Bank Marketing Campaign Classification
This repository contains a Python script to classify the success of a bank marketing campaign using a Decision Tree classifier. The dataset used in this project is the "Bank Marketing" dataset, which contains information on various marketing campaigns and whether or not the client subscribed to a term deposit.

Project Overview
The goal of this project is to build a model that can predict whether a client will subscribe to a term deposit based on various features such as age, job, marital status, and more. The dataset is preprocessed, and a Decision Tree classifier is trained and evaluated on the data.

Dataset
The dataset used is bank-full.csv, which can be found in the UCI Machine Learning Repository. It contains 45,211 records with 17 features.

Requirements
Python 3.x
Pandas
Scikit-learn
You can install the necessary packages using:


pip install pandas scikit-learn
Code Explanation
Data Loading: The dataset is loaded into a Pandas DataFrame.

Preprocessing:

Categorical features are encoded using LabelEncoder.
The dataset is split into features (X) and the target variable (y).
The data is further split into training and testing sets.
Feature scaling is applied using StandardScaler.
Model Training: A Decision Tree classifier is built and trained on the training data.

Model Evaluation:

Predictions are made on the test data.
The model's performance is evaluated using accuracy, confusion matrix, and classification report.
Usage
Place the bank-full.csv file in the appropriate directory.

Run the script to load the dataset, preprocess the data, train the model, and evaluate its performance:


python bank_marketing_classification.py
The script will output the model's accuracy, confusion matrix, and classification report.
Results:
The accuracy, and classification report will be printed to the console, providing insight into the model's performance.
