#imorting required libraries
import pickle
import rel
import feature_vectors
from sklearn.externals import joblib
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Taking input question and its body

inputQuestion = input("\nWelcome to StackOverFlow Answer Predictor! Please enter a technical question ->\n")
inputQuestionBody = input("Please enter some more details (a paragraph! as a question body)->\n")


with open('dataset', 'rb') as fp:
    dataset = pickle.load(fp)
#print('dataset loaded')

for myDict in dataset: #checking if queried question already in our dataset
    if myDict['qTitle'] == inputQuestion:
        if myDict['acceptAnsID'] == myDict['answerId']:
            print('\nAnswer found\nANSWER::')
            print(myDict['answerBody'])
            exit(0)
#print('not found in dataset')

#finding relevant question pool
rq = rel.RelevantQuestions()
relQIds = rq.basedOnTitle(inputQuestion)
#relQIds2 = rq.basedOnBody(inputQuestion)
relQIds.append(rq.basedOnBody(inputQuestionBody))
#print(relQIds)


#getting answers to corresponding questions
answerSet1 = []
labels_test1 = []
#answerSet2 = []
#labels_test2 = []
#print(relQIds)
if relQIds is None: #and relQIds2 is None:
    print("\n \nOops! I couldn't get a answer. Please try again!. Inconvenience caused is deeply regretted.\n")
    exit(0) 

for qId in relQIds:
    for myDict in dataset:
        if myDict['qId'] == qId:
            #print("Relevant Question :- ",myDict['qTitle'])
            answerSet1.append(myDict)
            if myDict['acceptAnsID'] == myDict['answerId']:
                labels_test1.append(1)
            else:
                labels_test1.append(0)
                
#print('answers fetched')

# get all feature vectors of answers
cf = feature_vectors.featureVectors()
features_test1 = cf.cfv(answerSet1)
#features_test2 = cf.cfv(answerSet2)
#print(features_test1)
#print('feature vectors obtained')

# getting all labels
clf = joblib.load('trained_3.pkl')
result1 = clf.predict(features_test1)
#result2 = clf.predict(features_test2)
#print(labels_test1)
#print(result1)

i = 0
k = 1
for label in result1:
    if label == 1:      
        print("\n \nAnswer %d:" %(k))
        print(answerSet1[i]['answerBody'])
        k = k + 1
    i = i + 1
    
i = 0
if k == 1:
    for label in labels_test1:
        if label == 1:       
            print("\n \nAnswer %d:" %(k))
            print(answerSet1[i]['answerBody'])
            k = k + 1
        i = i + 1

#end main 
