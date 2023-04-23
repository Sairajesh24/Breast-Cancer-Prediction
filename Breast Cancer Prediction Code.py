# importing Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import warnings
# Load the dataset
df = pd.read_csv("data.csv")

# Drop unnecessary columns
df.drop(df.columns[[-1,0]], axis=1, inplace=True)

# Map target variable to binary values
diag_map = {'M' : 1,'B' : 0}
df['diagnosis'] = df['diagnosis'].map(diag_map)

# Split into features and target
X = df[['radius_mean','perimeter_mean','area_mean','concavity_mean','concave points_mean']]
y = df['diagnosis'].values.ravel()

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model-KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_y_pred = knn.predict(X_test)
print("KNN Accuracy:", accuracy_score(knn_y_pred, y_test))

# Model-Logistic Regression
lr = LogisticRegression(random_state=0) 
lr.fit(X_train, y_train)
lr_y_pred = lr.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(lr_y_pred, y_test))

# Model-Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb_y_pred = gnb.predict(X_test)
print("Naive Bayes Accuracy:", accuracy_score(gnb_y_pred, y_test))

# K Fold Cross Validation
accuracy_all = []
cvs_all = []
warnings.filterwarnings("ignore")

# KNN
knn_scores = cross_val_score(knn, X, y, cv=10)
accuracy_all.append(accuracy_score(knn_y_pred, y_test))
cvs_all.append(np.mean(knn_scores))
print("KNN Accuracy: {0:.2%}".format(accuracy_score(knn_y_pred, y_test)))
print("KNN 10-Fold Cross Validation Score: {0:.2%} (+/- {1:.2%})".format(np.mean(knn_scores), np.std(knn_scores)*2))

# Logistic Regression
lr_scores = cross_val_score(lr, X, y, cv=10)
accuracy_all.append(accuracy_score(lr_y_pred, y_test))
cvs_all.append(np.mean(lr_scores))
print("Logistic Regression Accuracy: {0:.2%}".format(accuracy_score(lr_y_pred, y_test)))
print("Logistic Regression 10-Fold Cross Validation Score: {0:.2%} (+/- {1:.2%})".format(np.mean(lr_scores), np.std(lr_scores)*2))

# Naive Bayes
gnb_scores = cross_val_score(gnb, X, y, cv=10)
accuracy_all.append(accuracy_score(gnb_y_pred, y_test))
cvs_all.append(np.mean(gnb_scores))
print("Naive Bayes Accuracy: {0:.2%}".format(accuracy_score(gnb_y_pred, y_test)))
print("Naive Bayes 10-Fold Cross Validation Score: {0:.2%} (+/- {1:.2%})".format(np.mean(gnb_scores), np.std(gnb_scores)*2))