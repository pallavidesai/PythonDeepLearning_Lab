from sklearn.datasets import load_digits
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

# Loading the load_digita() dataset
loaddigitsdataset = load_digits()
plt.hist(loaddigitsdataset.target)
plt.show()

# getting the data and response of the dataset
x = loaddigitsdataset.data
y = loaddigitsdataset.target

# split the data for training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Using Naive Bayes to create a model
model = GaussianNB()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(metrics.accuracy_score(y_test, y_pred))