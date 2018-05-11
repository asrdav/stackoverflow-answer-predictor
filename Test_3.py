import pickle
from sklearn import tree
import numpy as np
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier

with open('features_test1', 'rb') as fp:
    features_test = pickle.load(fp)
with open('labels_test1', 'rb') as fp:
    labels_test = pickle.load(fp)

X_test = np.array(features_test)
y_test = np.array(labels_test)

print(len(X_test))
print(len(y_test))

clf = RandomForestClassifier()
clf = joblib.load('trained_3.pkl')

clf.predict(X_test)

res = clf.score(X_test,y_test)
print(res)
print("Testing Successful!")
