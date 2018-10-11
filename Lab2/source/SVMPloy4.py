# use the SVM with RBF kernel on the same dataset.
from sklearn.svm import SVC
from sklearn import datasets
from sklearn import datasets, metrics
from sklearn.cross_validation import train_test_split
from sklearn import neighbors, datasets
# Loading the dataset
digitsdataset = datasets.load_digits()
# getting the data and response of the dataset
x = digitsdataset.data
y = digitsdataset.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# Creating the Model
model = SVC(kernel="poly", degree=4, C=1, gamma=0.1)
model.fit(x_train, y_train)
# Do cross validation now
y_pred = model.predict(x_test)
print(metrics.accuracy_score(y_test, y_pred))


