from gensim.corpora import Dictionary
import pickle
from nltk.tokenize import word_tokenize
import csv

data=[]

with open('d1', 'rb') as fp:
    dataset = pickle.load(fp)
print('dataset loaded')

for mydict in dataset:
	query=[[w.lower() for w in word_tokenize(mydict['qTitle'])]]
	dct=Dictionary(query)
	data.append(dct)

with open('d2', 'wb') as fp:
    pickle.dump(data, fp)




