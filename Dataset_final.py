#preprocessing generated dataset from dataset.stackexchange
#in format suitable for feature extraction

import csv #importing necessary libraries
import re
import pickle

dataset = []

def cleanHTML(raw_html): #function to remove HTML chars from the raw dataset
    cleanr = re.compile('<.*?>')
    cleanText = re.sub(cleanr, '', raw_html) #removing html chars with null char
    return cleanText


def processTags(qtags):
    charToRemove = "<>" #removing angular brackets from question tags
    for char in qtags:
        if char in charToRemove:
            qtags = qtags.replace(char, ' ')
    return qtags


count = 0 #transforming raw dataset in usable dataset
#count1 = 0
count2 = 0
with open("/home/btech/siddharth.cs16/RawDataSet/QueryResultsF.csv", encoding="UTF-8") as f_obj:
    csvFile = csv.reader(f_obj)
    for line in csvFile:
        try:
            qId = int(line[0]) #following info we have
            #count2+=1          count2 is the no. of questions with Ids i.e. the total no. of questions.
            acceptAnsId = int(line[1])
            qScore = int(line[3])
            qViewCount = int(line[4])
            qBody = cleanHTML(line[6])
            qTitle = cleanHTML(line[7])
            qTags = processTags(line[8])
            answerId = int(line[13])
            answerScore = int(line[14])
            answerBody = cleanHTML(line[17])
            answerCommCount = int(line[18])
            answererReputation = int(line[22])
            answererUpvotes = int(line[23])
            answererDownvotes = int(line[24])
            if acceptAnsId < 1500001:    #acceptAnsId may or may not be below 1500001
                dataset.append(
                    {'qId': qId, 'acceptAnsID': acceptAnsId, 'qScore': qScore, 'qViewCount': qViewCount,
                     'qBody': qBody, 'qTitle': qTitle, 'qTags': qTags, 'answerId': answerId,
                     'answerScore': answerScore, 'answerBody': answerBody, 'answerCommCount': answerCommCount,
                     'answererReputation': answererReputation, 'answererUpvotes': answererUpvotes,
                     'answererDownvotes': answererDownvotes})
            count = count + 1   #count is the no. of questions with acceptedAnsId not being null and not necessarily below 1500001
            if count % 10000 == 0:
                print(count)
        except ValueError:
            pass

#print(count)
dataset = sorted(dataset, key=lambda k: k['qId'])

print('\nDataset Sorted\n')
#print(count2)

with open('dataset', 'wb') as fp: #dumping created dataset
    pickle.dump(dataset, fp)

print(len(dataset))        #len(dataset) is the dataset-length satisfying both non-empty criteria and acceptedAnsId criteria.
#print("QCount = ",count1)
print('\nDataset Preprocessing Successful\n')
#end of PA_Dataset.py
