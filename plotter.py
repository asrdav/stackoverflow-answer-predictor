import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.svm import SVC
import numpy as np
from sklearn.externals import joblib
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

with open('features_train', 'rb') as fp:
    features_train = pickle.load(fp)
with open('labels_train', 'rb') as fp:
    labels_train = pickle.load(fp)
with open('features_test', 'rb') as fp:
    features_test = pickle.load(fp)
with open('labels_test', 'rb') as fp:
    labels_test = pickle.load(fp)

X_train = np.array(features_train)
y_train = np.array(labels_train)
X_test = np.array(features_test)
y_test = np.array(labels_test)

'''clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
clf.fit(X_train,y_train)
clf.predict(X_test)
res_SVC = clf.score(X_test,y_test)*100'''

clf = tree.DecisionTreeClassifier()
clf.fit(X_train,y_train)
clf.predict(X_test)
res_DTC = clf.score(X_test,y_test)*100

clf = RandomForestClassifier()
clf.fit(X_train,y_train)
clf.predict(X_test)
res_RFC = clf.score(X_test,y_test)*100

clf = KNeighborsClassifier()
clf.fit(X_train,y_train)
clf.predict(X_test)
res_KNNC = clf.score(X_test,y_test)*100

clf = GaussianNB()
clf.fit(X_train,y_train)
clf.predict(X_test)
res_NBC = clf.score(X_test,y_test)*100

names = ['KNN' ,'DT', 'RF', 'NB']
values = [res_KNNC ,res_DTC, res_RFC, res_NBC]

plt.figure(None, figsize=(20, 10), dpi = 200)
plt.ylabel("Score")
plt.subplot(131)
plt.bar(names, values)
plt.subplot(133)
plt.scatter(names, values)
plt.subplot(133)
plt.plot(names, values)
plt.suptitle('Comparison of scores of different Algorithms')
plt.grid(True)
plt.show()
