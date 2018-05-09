import pickle
from sklearn.svm import SVC
import numpy as np
from sklearn.externals import joblib


with open('features_train', 'rb') as fp:
    features_train = pickle.load(fp)
with open('labels_train', 'rb') as fp:
    labels_train = pickle.load(fp)

x = np.array(features_train)
y = np.array(labels_train)
clf = SVC(kernel='rbf')
clf.fit(x,y)

joblib.dump(clf,'trained.pkl')
print('successful')

res = clf.score(x,y)
print(res)
