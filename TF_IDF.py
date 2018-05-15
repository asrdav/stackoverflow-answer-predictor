import gensim
import pickle
from nltk.tokenize import word_tokenize

with open('dataset', 'rb') as fp:
    data = pickle.load(fp)

raw_documents_qtitle = []
raw_documents_qbody = []

for myDict in data:
	raw_documents_qtitle.append(myDict['qTitle'])
	#raw_documents_qbody.append(myDict['qBody'])

#with open('data_qtitle', 'wb') as fp:
#    pickle.dump(raw_documents_qtitle, fp)
#with open('data_qbody', 'wb') as fp:
#    pickle.dump(raw_documents_qbody, fp)


gen_docs_qtitle = [[w.lower() for w in word_tokenize(text)] for text in raw_documents_qtitle]
#gen_docs_qbody = [[w.lower() for w in word_tokenize(text)] for text in raw_documents_qbody]

dictionary_qtitle = gensim.corpora.Dictionary(gen_docs_qtitle) #creating dictionary of tokens in questions
#dictionary_qbody = gensim.corpora.Dictionary(gen_docs_qbody) #creating dictionary of tokens in questions

with open('titleDictionary1', 'wb') as fp:
    pickle.dump(dictionary_qtitle, fp)
    
#with open('bodyDictionary1', 'wb') as fp:
#   pickle.dump(dictionary_qbody, fp)


corpus_qtitle = [dictionary_qtitle.doc2bow(gen_doc) for gen_doc in gen_docs_qtitle] #creating bag of words(corpus) of text
#corpus_qbody = [dictionary_qbody.doc2bow(gen_doc) for gen_doc in gen_docs_qbody] #creating bag of words(corpus) of text


tf_idf_qtitle = gensim.models.TfidfModel(corpus_qtitle) #creating tfidf vector of text
#tf_idf_qbody = gensim.models.TfidfModel(corpus_qbody) #creating tfidf vector of text
with open('titleTfIdf1', 'wb') as fp:
    pickle.dump(tf_idf_qtitle, fp)
    
#with open('bodyTfIdf1', 'wb') as fp:
#   pickle.dump(tf_idf_qbody, fp)


sims_qtitle = gensim.similarities.Similarity('/home/btech/siddharth.cs16/perl5',tf_idf_qtitle[corpus_qtitle],num_features=len(dictionary_qtitle))

#sims_qbody = gensim.similarities.Similarity('/home/btech/siddharth.cs16/perl5',tf_idf_qbody[corpus_qbody],num_features=len(dictionary_qbody))

with open('titleSimilarity1', 'wb') as fp:
    pickle.dump(sims_qtitle, fp)
    
#with open('bodySimilarity1', 'wb') as fp:
#    pickle.dump(sims_qbody, fp)
