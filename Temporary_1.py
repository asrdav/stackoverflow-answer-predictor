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
          qBody = cleanHTML(line[6])
          qTitle = cleanHTML(line[7])
          qTags = processTags(line[8])
          if acceptAnsId < 900001:
            data.append(
              {'qId': qId, 'acceptAnsID': acceptAnsId, 'qBody': qBody, 'qTitle': qTitle, 'qTags': qTags})
          count = count + 1
          if count % 10000 == 0:
            print(count)
        except ValueError:
              pass

data = sorted(data, key=lambda k: k['qId'])

print('data sorted')

with open('d1', 'wb') as fp:
    pickle.dump(data, fp)


     
                
               
