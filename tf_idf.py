import gensim
import pickle
from nltk.tokenize import word_tokenize
#print(dir(gensim))

with open('dataset', 'rb') as fp:
    data = pickle.load(fp)
    
with open('data', 'wb') as fp:
    pickle.dump(data, fp)

raw_documents = []
for myDict in data:
	raw_documents.append(myDict['qTitle'])
#print("Number of documents:",len(raw_documents))


gen_docs = [[w.lower() for w in word_tokenize(text)] for text in raw_documents]
#print(gen_docs)

dictionary = gensim.corpora.Dictionary(gen_docs)
with open('titleDictionary', 'wb') as fp:
    pickle.dump(dictionary, fp)
'''print(dictionary[5])
print(dictionary.token2id['road'])
print("Number of words in dictionary:",len(dictionary))
for i in range(len(dictionary)):
    print(i, dictionary[i])'''

corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
#print(corpus)

tf_idf = gensim.models.TfidfModel(corpus)
with open('titleTfIdf', 'wb') as fp:
    pickle.dump(tf_idf, fp)
'''print(tf_idf)
s = 0
for i in corpus:
    s += len(i)
print(s)'''

sims = gensim.similarities.Similarity('/home/abhisid/Documents/',tf_idf[corpus],num_features=len(dictionary))
with open('titleSimilarity', 'wb') as fp:
    pickle.dump(sims, fp)
#print(sims)
#print(type(sims))
