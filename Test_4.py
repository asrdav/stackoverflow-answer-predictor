import pickle
from sklearn import tree
import numpy as np
from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB

with open('features_test', 'rb') as fp:
    features_test = pickle.load(fp)
with open('labels_test', 'rb') as fp:
    labels_test = pickle.load(fp)

X_test = np.array(features_test)
y_test = np.array(labels_test)

print(len(X_test))
print(len(y_test))

clf = GaussianNB()
clf = joblib.load('trained_4.pkl')

clf.predict(X_test)

res = clf.score(X_test,y_test)
print(res)
print("Testing Successful!")
