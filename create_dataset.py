import csv
import re
import pickle

data = []


def cleanHTML(raw_html):
    cleanr = re.compile('<.*?>')
    cleanText = re.sub(cleanr, '', raw_html)
    return cleanText


def processTags(qtags):
    charToRemove = "<>"
    for char in qtags:
        if char in charToRemove:
            qtags = qtags.replace(char, ' ')
    return qtags


count = 0
with open("/home/abhisid/Documents/Queries.csv", encoding="UTF-8") as f_obj:
    csvFile = csv.reader(f_obj)
    for line in csvFile:
        try:
            qId = int(line[0])
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
            if acceptAnsId < 900001:
                data.append(
                    {'qId': qId, 'acceptAnsID': acceptAnsId, 'qScore': qScore, 'qViewCount': qViewCount,
                     'qBody': qBody, 'qTitle': qTitle, 'qTags': qTags, 'answerId': answerId,
                     'answerScore': answerScore, 'answerBody': answerBody, 'answerCommCount': answerCommCount,
                     'answererReputation': answererReputation, 'answererUpvotes': answererUpvotes,
                     'answererDownvotes': answererDownvotes})
            count = count + 1
            if count % 10000 == 0:
                print(count)
        except ValueError:
            pass


data = sorted(data, key=lambda k: k['qId'])

print('data sorted')

with open('dataset', 'wb') as fp:
    pickle.dump(data, fp)

print(len(data))
print('Successful')
