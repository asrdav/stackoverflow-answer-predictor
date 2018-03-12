from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from collections import Counter
from enchant.checker import SpellChecker
from textblob import TextBlob
from rake_nltk import Rake


class TextProperties:
    @staticmethod
    def NonstopWords(text):
        text = re.sub(r'[^\w\s]', ' ', text)
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        filtered_sentence = []
        for w in word_tokens:
            if w not in stop_words:
                filtered_sentence.append(w)
        return len(filtered_sentence)

    @staticmethod
    def UniqueWords(text):
        text = re.sub(r'[^\w\s]', ' ', text)
        c = Counter(text.split())
        c = set(c)
        return len(c)

    @staticmethod
    def spellcheck(text):
        chkr = SpellChecker("en_US")
        chkr.set_text(text)
        c = 0
        for _ in chkr:
            c = c + 1
        return c

    @staticmethod
    def subjective(text):
        sub = TextBlob(text).sentiment.subjectivity
        return sub

    @staticmethod
    def relevancy(answer, qBody, qTags):
        r = Rake()
        r.extract_keywords_from_text(answer)
        keyfromans = r.get_ranked_phrases()
        r.extract_keywords_from_text(qBody)
        keyfromque = r.get_ranked_phrases()
        value = 100
        try:
            for anskey in keyfromans:
                for quekey in keyfromque:
                    if anskey == quekey:
                        value = value + 100
            for anskey in keyfromans:
                if anskey in qTags:
                    value = value + 40
        except:
            value = 0
        return value
