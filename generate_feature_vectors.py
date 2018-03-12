import pickle
import create_feature_vectors

with open('dataset', 'rb') as fp:
    data = pickle.load(fp)

print('data loaded')
count = 0
distinctqID = []
for myDict in data:
    if myDict['qId'] not in distinctqID:
        distinctqID.append(myDict['qId'])
    count = count + 1
    if count > 200000:
        break

print('distinct')
features_train = []
labels_train = []
fv = create_feature_vectors.featureVectors()

features_test = []
labels_test = []

count = 0
print('started')
for x in distinctqID:
    if count > 10000:
        indata = []
        for myDict in data:
            if myDict['qId'] > x:
                break
            if myDict['qId'] != x:
                continue
            indata.append(myDict)
            if myDict['acceptAnsID'] == myDict['answerId']:
                labels_test.append(1)
            else:
                labels_test.append(0)
        fvs = fv.cfv(indata)
        for vector in fvs:
            features_test.append(vector)
        count = count + 1
        if count > 15000:
            break
    else:
        indata = []
        for myDict in data:
            if myDict['qId'] > x:
                break
            if myDict['qId'] != x:
                continue
            indata.append(myDict)
            if myDict['acceptAnsID'] == myDict['answerId']:
                labels_train.append(1)
            else:
                labels_train.append(0)
        fvs = fv.cfv(indata)
        for vector in fvs:
            features_train.append(vector)
        count = count + 1

    if count%100 == 0:
        print(count)

with open('features_train', 'wb') as fp:
    pickle.dump(features_train, fp)
with open('labels_train', 'wb') as fp:
    pickle.dump(labels_train, fp)
with open('features_test', 'wb') as fp:
    pickle.dump(features_test, fp)
with open('labels_test', 'wb') as fp:
    pickle.dump(labels_test, fp)

print('successful')
