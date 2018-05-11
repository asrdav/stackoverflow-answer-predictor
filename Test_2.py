import pickle
from sklearn import tree
import numpy as np
from sklearn.externals import joblib


with open('features_test1', 'rb') as fp:
    features_test = pickle.load(fp)
with open('labels_test1', 'rb') as fp:
    labels_test = pickle.load(fp)

print(len(features_test))
print(len(labels_test))

X_test = np.array(features_test)
y_test = np.array(labels_test)

clf = tree.DecisionTreeClassifier()
clf = joblib.load('trained_2.pkl')

res = clf.predict(X_test)

#print(res.tolist().count(1))
#print(labels_test.count(1))
print(clf.score(X_test,y_test))
print("Testing Successful!")
