from sklearn import neighbors, datasets
from sklearn import model_selection, metrics
from sklearn.cross_validation import train_test_split
#  now load the dataset
irisdataset = datasets.load_iris()
x = irisdataset.data
y = irisdataset.target
# train and test of data
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=22, test_size=0.2)
# for k=1 neighbors
model = neighbors.KNeighborsClassifier(n_neighbors=1)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
# prints accuracy
print("Accuracy when k=1:", metrics.accuracy_score(y_test, y_pred))
# for k=50 neighbors
model = neighbors.KNeighborsClassifier(n_neighbors=50)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
# prints accuracy
print("Accuracy when k=50:", metrics.accuracy_score(y_test, y_pred))