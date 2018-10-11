# use the SVM with RBF kernel on the same dataset.
from sklearn.svm import SVC
from sklearn import datasets
from sklearn import datasets, metrics
from sklearn.cross_validation import train_test_split
# Loading the dataset
irisdataset = datasets.load_iris()
# getting the data and response of the dataset
x = irisdataset.data
y = irisdataset.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# Creating the Model
model = SVC(kernel='rbf', C=1, gamma=0.1)
model.fit(x_train, y_train)
# Do cross validation now
y_pred = model.predict(x_test)
print(metrics.accuracy_score(y_test, y_pred))