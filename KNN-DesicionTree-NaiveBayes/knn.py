import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class KNN:
    def __init__(self, k):
        self.x_train, self.y_train = pd.DataFrame(), pd.DataFrame()
        self.k=k

    def distance(self, x1, x2):
        np_x1 = np.array(x1)
        np_x2 = np.array(x2)
        return np.linalg.norm(np_x1 - np_x2)

    def train(self, x, yn):
        self.x_train = x
        self.y_train = y

    def predict(self, X: pd.DataFrame):
        y_pred = []
        for j in range(len(X)):
            x = X.iloc[j]
            distances = []
            for i in range(len(self.x_train)):
                instance = self.x_train.iloc[i]
                cls = self.y_train.iloc[i]

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

if __name__ == '__main__':
    df = pd.read_csv('weather_forecast_data.csv')

    # Split the data into training and testing sets (80% for training, 20% for testing)
    X = df.drop(columns=['Rain'])  # Features
    y = df['Rain']  # Target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.02, random_state=42)

    knn = KNN(3)
    knn.train(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(y_pred)
