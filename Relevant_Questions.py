#importing important libraries
import pickle
from nltk.tokenize import word_tokenize

resQId = []
resQId1 = []
class RelevantQuestions:

    
    @staticmethod
    def basedOnTitle(question): #getting most similar questions pool
        
        with open('dataset', 'rb') as fp:
            data = pickle.load(fp)
        with open('titleSimilarity1', 'rb') as fp:
            sims = pickle.load(fp)
        with open('titleDictionary1', 'rb') as fp:
            dictionary = pickle.load(fp)
        with open('titleTfIdf1', 'rb') as fp:
            tf_idf = pickle.load(fp)
        
        query_doc = [w.lower() for w in word_tokenize(question)] #tokenizing question
        query_doc_bow = dictionary.doc2bow(query_doc) #converting it to bag of words representation
        query_doc_tf_idf = tf_idf[query_doc_bow]
        res = sims[query_doc_tf_idf] #querying for most simillar question
        
	#here we take 2 most similar questions
        idx1 = 0
        idx2 = 0
        maxSim = 0
        for i in range(0, len(res)):
            if res[i] > maxSim:
                idx1 = i
                maxSim = res[i]
        
        maxSim = 0
        for i in range(0, len(res)):
            if res[i] > maxSim and i != idx1:
                idx2 = i
                maxSim = res[i]
        resQId.append(data[idx1]['qId'])
        if data[idx1]['qId'] != data[idx2]['qId']:
            resQId.append(data[idx2]['qId'])
        return resQId
    
    @staticmethod
    def basedOnBody(question): #getting most similar questions pool
        
        with open('dataset', 'rb') as fp:
            data = pickle.load(fp)
        with open('bodySimilarity1', 'rb') as fp:
            sims = pickle.load(fp)
        with open('bodyDictionary1', 'rb') as fp:
            dictionary = pickle.load(fp)
        with open('bodyTfIdf1', 'rb') as fp:
            tf_idf = pickle.load(fp)
        
        query_doc = [w.lower() for w in word_tokenize(question)] #tokenizing question
        query_doc_bow = dictionary.doc2bow(query_doc) #converting it to bag of words representation
        query_doc_tf_idf = tf_idf[query_doc_bow]
        res = sims[query_doc_tf_idf] #querying for most simillar question
        
	#here we take 2 most similar questions
        idx3 = 0
        idx4 = 0
        maxSim = 0
        for i in range(0, len(res)):
            if res[i] > maxSim:
                idx3 = i
                maxSim = res[i]
        
        maxSim = 0
        for i in range(0, len(res)):
            if res[i] > maxSim and i != idx3:
                idx4 = i
                maxSim = res[i] 
        resQId1.append(data[idx3]['qId'])
        if data[idx3]['qId'] != data[idx4]['qId']:
            resQId1.append(data[idx4]['qId'])  
        return resQId1
  
