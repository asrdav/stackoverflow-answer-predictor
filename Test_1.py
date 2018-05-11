import pickle
from sklearn.svm import SVC
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

clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
clf = joblib.load('trained_1.pkl')

#print(res.tolist().count(1))
#print(labels_test.count(1))
print(clf.score(X_test,y_test))
print("Testing Successful!")
