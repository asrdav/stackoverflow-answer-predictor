import readability #importing necessary libraries
import info
import properties


class featureVectors:
    @staticmethod
    def assignValue(myList):
        list2 = sorted(myList, reverse=True)
        values = []
        const = 5040 #normalization factor
        n = len(myList)
        for value1 in myList:
            for i in range(0, n):
                if value1 == list2[i]:
                    values.append((const / n) * (n - i))
                    break
        return values

    def getInformation(self, listOfDicts): #getting information provided by the answer body using markov model and entropy
        inf = info.Informativity()
        listOfInformation = []
        for row in listOfDicts:
            totalEntropy = 0
            model, stats = inf.markov_model(inf.chars(row['answerBody']), 3)
            for prefix in stats:
                totalEntropy = totalEntropy + inf.entropy(stats, stats[prefix])
            listOfInformation.append(abs(totalEntropy))
        return self.assignValue(listOfInformation)

    def getRelevancy(self, listOfDicts): #getting relevancy between answerBody, qBody, qTags
        tp = properties.TextProperties()
        listOfRel = []
        for row in listOfDicts:
            listOfRel.append(tp.relevancy(row['answerBody'], row['qBody'], row['qTags']))
        return self.assignValue(listOfRel)

    def getUniqueWords(self, listOfDicts): #getting unique words in answerBody
        tp = properties.TextProperties()
        listOfUw = []
        for row in listOfDicts:
            listOfUw.append(tp.UniqueWords(row['answerBody']))
        return self.assignValue(listOfUw)

    def getNSWords(self, listOfDicts): #getting Nonstop-Words in answerBody
        tp = properties.TextProperties()
        listOfNs = []
        for row in listOfDicts:
            listOfNs.append(tp.NonstopWords(row['answerBody']))
        return self.assignValue(listOfNs)

    def getSub(self, listOfDicts): #getting subjectivity of answerBody
        tp = properties.TextProperties()
        listOfSub = []
        for row in listOfDicts:
            listOfSub.append(tp.subjective(row['answerBody']))
        return self.assignValue(listOfSub)

    def getAS(self, listOfDicts): #using answerScore as feature
        listOfAs = []
        for row in listOfDicts:
            listOfAs.append(row['answerScore'])
        return self.assignValue(listOfAs)

    def getDV(self, listOfDicts): #using negation of number of answererDownvotes as a feature
        listOfDV = []
        for row in listOfDicts:
            listOfDV.append(row['answererDownvotes'] * -1)
        return self.assignValue(listOfDV)

    def getUV(self, listOfDicts): #using number of answererUpvotes as a feature
        listOfUV = []
        for row in listOfDicts:
            listOfUV.append(row['answererUpvotes'])
        return self.assignValue(listOfUV)

    def getComm(self, listOfDicts): #using number of comments on the answer as a feature
        listOfComm = []
        for row in listOfDicts:
            listOfComm.append(row['answerCommCount'])
        return self.assignValue(listOfComm)

    def getRead(self, listOfDicts): #using readabilty index of answerBody as a feature
        listOfRead = []
        ts = readability.TextStatistics()
        for row in listOfDicts:
            listOfRead.append(ts.text_standard(row['answerBody']))
        return self.assignValue(listOfRead)

    def getDW(self, listOfDicts): #using difficult words in answerBody as a feature
        listOfDW = []
        ts = readability.TextStatistics()
        for row in listOfDicts:
            listOfDW.append(ts.difficult_words(row['answerBody']))
        return self.assignValue(listOfDW)

    def getRepu(self, listOfDicts): #using answerer reputation as a feature 
        listOfRepu = []
        for row in listOfDicts:
            listOfRepu.append(row['answererReputation'])
        return self.assignValue(listOfRepu)

    def cfv(self, listOfDicts): #to create feature vector
        n = len(listOfDicts)
        features = [[] for _ in range(n)]

        infor = self.getInformation(listOfDicts) #1
        for i in range(0, n):
            features[i].append(infor[i])

        rel = self.getRelevancy(listOfDicts) #2
        for i in range(0, n):
            features[i].append(rel[i])

        uw = self.getUniqueWords(listOfDicts) #3
        for i in range(0, n):
            features[i].append(uw[i])

        ns = self.getNSWords(listOfDicts) #4
        for i in range(0, n):
            features[i].append(ns[i])

        sub = self.getSub(listOfDicts) #5
        for i in range(0, n):
            features[i].append(sub[i])

        answerScore = self.getAS(listOfDicts) #6
        for i in range(0, n):
            features[i].append(answerScore[i])

        downVotes = self.getDV(listOfDicts) #7
        for i in range(0, n):
            features[i].append(downVotes[i])

        upVotes = self.getUV(listOfDicts) #8
        for i in range(0, n):
            features[i].append(upVotes[i])

        comments = self.getComm(listOfDicts) #9
        for i in range(0, n):
            features[i].append(comments[i])

        readable = self.getRead(listOfDicts) #10
        for i in range(0, n):
            features[i].append(readable[i])

        diffWords = self.getDW(listOfDicts) #11
        for i in range(0, n):
            features[i].append(diffWords[i])

        repu = self.getRepu(listOfDicts) #12
        for i in range(0, n):
            features[i].append(repu[i])

        return features
