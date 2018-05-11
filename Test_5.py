import pickle
import numpy as np
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier

with open('features_test', 'rb') as fp:
    features_test = pickle.load(fp)
with open('labels_test', 'rb') as fp:
    labels_test = pickle.load(fp)

X_test = np.array(features_test)
y_test = np.array(labels_test)

print(len(X_test))
print(len(y_test))

clf = KNeighborsClassifier()
clf = joblib.load('trained_5.pkl')

clf.predict(X_test)

res = clf.score(X_test,y_test)
print(res)
print("Testing Successful!")
