#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score


# In[79]:


# load the dataset
df = pd.read_csv("weather_forecast_data.csv")


# In[ ]:


# Task 1: Preprocessing
# 1. Check for missing values
def checkMissing(df):
    print("Missing values per column:")
    print(df.isnull().sum())


# In[ ]:


# 2. Handle missing values by two techniques
def handle_missing(df, strategy):
    if strategy == "drop":
        return df.dropna()
    elif strategy == "replace":
        df_copy = df.copy()
        # Fill numeric columns with mean
        for col in df_copy.select_dtypes(include="number").columns:
            df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
        # Fill categorical columns with mode
        for col in df_copy.select_dtypes(include="object").columns:
            df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0])
        return df_copy
    else:
        raise ValueError("Invalid missing value strategy")


# In[ ]:


# 3. Preprocessing
def preprocess(df, scaling):
    X = df.drop(columns=["Rain"])
    y = df["Rain"]

    # Encode target
    y = LabelEncoder().fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale numeric features
    scaler = MinMaxScaler() if scaling == "min-max" else StandardScaler()
    toscale = X_train.select_dtypes(include="number").columns
    X_train[toscale] = scaler.fit_transform(X_train[toscale])
    X_test[toscale] = scaler.transform(X_test[toscale])

    return X_train, X_test, y_train, y_test


# In[ ]:


class KNN:
    def __init__(self, k):
        self.x_train, self.y_train = pd.DataFrame(), pd.DataFrame()
        self.k=k

    def distance(self, x1, x2):
        np_x1 = np.array(x1)
        np_x2 = np.array(x2)
        return np.linalg.norm(np_x1 - np_x2)

    def train(self, x, y):
        self.x_train = x
        self.y_train = y

    def predict(self, X: pd.DataFrame):
        y_pred = []
        for j in range(len(X)):
            x = X.iloc[j]
            distances = []
            for i in range(len(self.x_train)):
                instance = self.x_train.iloc[i]
                cls = self.y_train[i]

                distance = self.distance(x, instance)
                distances.append((distance, cls))

            sorted_distances = sorted(distances, key=lambda x: x[0])
            freq = dict()
            count = 0
            group = 0
            for i in range(self.k):
                if sorted_distances[i][1] not in freq:
                    freq[sorted_distances[i][1]] = 1
                else:
                    freq[sorted_distances[i][1]] += 1
                if freq[sorted_distances[i][1]] > count:
                    count = freq[sorted_distances[i][1]]
                    group = sorted_distances[i][1]
            y_pred.append(group)

        return y_pred


# In[ ]:


# 4. Evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    return accuracy, precision, recall


# In[ ]:


for i in range(3, 12, 2):
    df = pd.read_csv("weather_forecast_data.csv")
    print("checking missing values for k = "+ i)
    checkMissing(df)
    df = handle_missing(df, "replace")
    X_train, X_test, y_train, y_test = preprocess(df,"min-max")
    knn = KNN(i)
    knn.train(X_train, y_train)
    accuracy, precision, recall = evaluate_model(knn,X_test, y_test)
    print("knn with k = " + str(i))
    print("Accuracy: " + str(accuracy))
    print("Precision: " + str(precision))
    print("recall: " + str(recall))






