import pickle #importing necessary libraries
import feature_vectors

with open('dataset', 'rb') as fp:
    data = pickle.load(fp)
print('data loaded')

count = 0
distinctqID = []
for myDict in data: #myDict : variable used to check for distinct questions
    if myDict['qId'] not in distinctqID:
        distinctqID.append(myDict['qId'])
        count = count + 1
    if count > 1500000: #variable count to limit no. of questions
        break
print('distinct')
print(count)

features_train = [] #features for training the model
labels_train = [] #labels for training the model 
features_test = [] #features for testing the model
labels_test = [] #labels for testing the model

class_FeatureVector = feature_vectors.featureVectors() #class_FeatureVector is an object of class


count = 0
print('\nGenerationg Feature Vectors started\n')

for x in distinctqID:
    if count > 88101: #creating testing set
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
                
        fvs = class_FeatureVector.cfv(indata)
        for vector in fvs:
            features_test.append(vector)
        count = count + 1
        if count > 117589: #limiting amount of testing data
            break
    else:
        indata = []
        for myDict in data:
            if myDict['qId'] > x:
                break
            if myDict['qId'] != x:
                continue
        
            indata.append(myDict) #appending dataset fields in indata
            if myDict['acceptAnsID'] == myDict['answerId']:
                labels_train.append(1) #if accepted answer id is equal to an answer id
            else:
                labels_train.append(0)
        fvs = class_FeatureVector.cfv(indata)
        for vector in fvs:
            features_train.append(vector) #creating training features
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

print('\nGenerating features vectors successful\n')
