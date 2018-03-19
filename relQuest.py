import pickle
from nltk.tokenize import word_tokenize


class RelevantQuestions:
    @staticmethod
    def basedOnTitle(question):
        with open('data', 'rb') as fp:
            data = pickle.load(fp)
        with open('titleSimilarity', 'rb') as fp:
            sims = pickle.load(fp)
        with open('titleDictionary', 'rb') as fp:
            dictionary = pickle.load(fp)
        with open('titleTfIdf', 'rb') as fp:
            tf_idf = pickle.load(fp)
        query_doc = [w.lower() for w in word_tokenize(question)]
        query_doc_bow = dictionary.doc2bow(query_doc)
        query_doc_tf_idf = tf_idf[query_doc_bow]
        res = sims[query_doc_tf_idf]
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

        resQId = [data[idx1]['qId'], data[idx2]['qId']]
        if(data[idx1]['qId']==data[idx2]['qId']):
            resQId=[data[idx1]['qId']]
        return resQId
