import pickle
import relQuest
import create_feature_vectors
from sklearn.externals import joblib
from sklearn import tree

# Taking input question and its body
#print("Enter the Question:")
inputQuestion = input("Enter the Question:")
# inputQuestionBody = "I want to know how the function are defined in python " \
#                    "and how the parameter passing is done."


with open('dataset', 'rb') as fp:
    dataset = pickle.load(fp)
print('dataset loaded')

for myDict in dataset:
    if myDict['qTitle'] == inputQuestion:
        if myDict['acceptAnsID'] == myDict['answerId']:
            print('Answer found')
            print(myDict['answerBody'])
            exit(0)
print('not found in dataset')

# Getting relevant question pool
rq = relQuest.RelevantQuestions()
relQIds = rq.basedOnTitle(inputQuestion)
# relQIds.append(rq.basedOnBody(inputQuestionBody))

#print(relQIds)


# Getting answers to corresponding questions
answerSet = []
labels_test = []
for qId in relQIds:
    for myDict in dataset:
        if myDict['qId'] == qId:
            #print(myDict['qTitle'])
            answerSet.append(myDict)
            if myDict['acceptAnsID'] == myDict['answerId']:
                labels_test.append(1)
            else:
                labels_test.append(0)
print('answers fetched')

# get all feature vectors of answers
cf = create_feature_vectors.featureVectors()
features_test = cf.cfv(answerSet)

#print(features_test)
print('feature vectors obtained')

# getting all labels
clf = joblib.load('trained2.pkl')
result = clf.predict(features_test)
'''print(labels_test)'''
print(result)


i = 0
k = 0
for label in result:
    if label == 1:
        print("Answer %d:"%(k+1))
        print(answerSet[i]['answerBody'])
        k=k+1
    i = i + 1
