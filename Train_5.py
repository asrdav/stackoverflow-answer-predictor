import pickle
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.externals import joblib


with open('features_train', 'rb') as fp:
    features_train = pickle.load(fp)
with open('labels_train', 'rb') as fp:
    labels_train = pickle.load(fp)

print(len(features_train))
print(len(labels_train))

X_train = np.array(features_train)
y_train = np.array(labels_train)

clf = KNeighborsClassifier()
clf.fit(X_train,y_train)

joblib.dump(clf,'trained_5.pkl')

res = clf.score(X_train,y_train)
print(res)
print('Training Successful!')

