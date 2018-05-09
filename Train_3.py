import pickle
from sklearn import tree
import numpy as np
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier

with open('features_train', 'rb') as fp:
    features_train = pickle.load(fp)
with open('labels_train', 'rb') as fp:
    labels_train = pickle.load(fp)

print(len(features_train))
print(len(labels_train))
x = np.array(features_train)
y = np.array(labels_train)
clf = RandomForestClassifier()
clf.fit(x,y,sample_weight=None)

joblib.dump(clf,'trained3.pkl')
print('successful')

res = clf.score(x,y)
print(res)
