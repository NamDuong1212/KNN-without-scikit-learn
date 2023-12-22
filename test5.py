import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

#khoảng cách Euclidean giữa hai điểm
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

#dự đoán nhãn của một điểm dữ liệu trong tập huấn luyện
def predict_label(X_train, y_train, x_test, k):
    distances = []
    for i in range(len(X_train)):
        distance = euclidean_distance(X_train[i], x_test)
        distances.append((distance, y_train[i]))

    distances.sort(key=lambda x: x[0])
    k_nearest_neighbors = distances[:k]

    #đếm số lượng nhãn và chọn nhãn phổ biến nhất
    labels = [neighbor[1] for neighbor in k_nearest_neighbors]
    predicted_label = max(set(labels), key=labels.count)
    
    return predicted_label

def knn_predict(X_train, y_train, X_test, k):
    predictions = []
    for i in range(len(X_test)):
        predicted_label = predict_label(X_train, y_train, X_test[i], k)
        predictions.append(predicted_label)
    return np.array(predictions)

heart_data = pd.read_csv('C:/Users/nguye/Documents/kaggle/input/heart-disease-uci/heart.csv')

y = heart_data["target"]
X = heart_data.drop('target', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

k_value = 5
custom_knn_predicted = knn_predict(X_train, y_train.values, X_test, k_value)

custom_knn_conf_matrix = confusion_matrix(y_test, custom_knn_predicted)
custom_knn_acc_score = accuracy_score(y_test, custom_knn_predicted)
print("Confusion matrix on testing set")
print(custom_knn_conf_matrix)
print("\n")
print("Accuracy of Custom K-NeighborsClassifier on testing set:", custom_knn_acc_score * 100, '\n')
print(classification_report(y_test, custom_knn_predicted))
