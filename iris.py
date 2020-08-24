import mglearn
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris_data = load_iris()

print(iris_data.keys())
print("-------------------------------")
print("target namse = {}".format(iris_data["target_names"]))
print("-------------------------------")
print(iris_data["target"])
print("-------------------------------")
print(iris_data["data"][:5])
print("-------------------------------")
print("feature_names: {}".format(iris_data["feature_names"]))

# here we split the data by 2. One for train One for test
X_train, X_test, y_train, y_test = train_test_split(iris_data["data"], iris_data["target"], random_state=0)


# here we created dataframe from X_train Numpy Array. And we added columns from iris_data
iris_dataframe = pd.DataFrame(X_train, columns=iris_data["feature_names"])

# create scatter matrix from the dataframe, color by y_train
grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)

#plt.show()

# Build our model on our training set.
# This object contains the algorithm that build the model and make predictions on the new data points.
knn = KNeighborsClassifier(n_neighbors=1)

# Here we call fit function
# Our model is created based on the parameters that we passed.
# One is X_train np array training data, the other one is y_train training label.
knn.fit(X_train, y_train)

X_new = np.array([[5.4 , 3., 4.5, 1.5]])

"""
print("data: {}".format(iris_data["data"][:5])) # Raw data 
print("X_train: {}".format(X_train[:5])) # Training data, shuffled
print("target: {}".format(iris_data["target"][:5])) # raw target
print("y_train: {}".format(y_train[:5])) # target training data
"""

prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("predicted target names: {}".format(iris_data["target_names"][prediction]))
print("--------------------------------------")

# here we predict X_test np array to measure the accuracy of the model.
y_pred = knn.predict(X_test)
print("test set prediction: {}".format(y_pred))
# Modelin tahminleri ve gerçek tahminlerimizin mean'i alınır.
print("test set score: {:.2f}".format(np.mean(y_pred == y_test)))
# Test datamızı ve test gerçek sonuçlarımızı knn.score'a sokuyoruz.
print("test set score: {:.2f}".format(knn.score(X_test, y_test)))

