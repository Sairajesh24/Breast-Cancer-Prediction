# Breast-Cancer-Prediction
This code uses several modules from the scikit-learn library for building and evaluating classification models. Here is a brief description of each module used in the code:

pandas: Used for data manipulation and analysis. 
numpy: Used for numerical computations and array manipulation. 
sklearn.model_selection.train_test_split: Used for splitting the dataset into training and testing sets. 
sklearn.neighbors.KNeighborsClassifier: Used for building K-Nearest Neighbors classification models. 
sklearn.metrics.accuracy_score: Used for calculating the accuracy of classification models. 
sklearn.linear_model.LogisticRegression: Used for building logistic regression classification models. 
sklearn.naive_bayes.GaussianNB: Used for building Gaussian Naive Bayes classification models. 
sklearn.model_selection.cross_val_score: Used for performing K-Fold cross-validation to evaluate model performance.

This code loads a breast cancer dataset from a CSV file, preprocesses the data by dropping unnecessary columns and converting categorical data to numerical data, splits the data into training and testing sets, builds three different classification models (K-Nearest Neighbors, Logistic Regression, and Gaussian Naive Bayes), evaluates the models using accuracy score and 10-fold cross-validation, and selects the best model based on the evaluation results. The objective of this project is to classify breast tumors as malignant or benign based on several features such as radius, perimeter, area, and concavity of the tumor.
