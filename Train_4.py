from sklearn.naive_bayes import GaussianNB
import pickle
import numpy as np
from sklearn.externals import joblib
# Create SVM classification object model = GaussianNB() # there is other distribution for multinomial classes like Bernoulli Naive Bayes, Refer link
# Train the model using the training sets and check score

with open('features_train1', 'rb') as fp:
    features_train = pickle.load(fp)
with open('labels_train1', 'rb') as fp:
    labels_train = pickle.load(fp)
    
print(len(features_train))
print(len(labels_train))
    
X_train = np.array(features_train)
y_train = np.array(labels_train)

clf = GaussianNB()
    
clf.fit(X_train, y_train)
res = clf.score(X_train,y_train)
joblib.dump(clf,'trained_4.pkl')

print(res)
print('Training Successful!')
