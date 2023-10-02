import re
import numpy as np
import pandas as pd
import hazm
import pymongo
import copy
import pickle
from django.core.cache import cache
from cleantext import clean
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from nltk import sent_tokenize
from sentence_transformers import SentenceTransformer
from bson.binary import Binary
from .models import Similarity
import os

import nltk
nltk.download('punkt')

connString = os.environ['MONGODB_CONNSTRING']


def cleanHtml(raw_html):
    cleanRegex = re.compile('<.*?>')
    cleanText = re.sub(cleanRegex, '', raw_html)
    return cleanText


def cleaning(text):
    text = text.strip()

    text = clean(text, fix_unicode=True, to_ascii=False, lower=True, no_line_breaks=True, no_urls=True, no_emails=True,
                 no_phone_numbers=True, no_numbers=False, no_digits=False, no_currency_symbols=True, no_punct=False,
                 replace_with_url="", replace_with_email="", replace_with_phone_number="", replace_with_number="",
                 replace_with_digit="0", replace_with_currency_symbol="")

    text = cleanHtml(text)

    normalizer = hazm.Normalizer()
    text = normalizer.normalize(text)

    weird_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u'\U00010000-\U0010ffff'
                               u"\u200d"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\u3030"
                               u"\ufe0f"
                               u"\u2069"
                               u"\u2066"
                               # u"\u200c"
                               u"\u2068"
                               u"\u2067"
                               "]+", flags=re.UNICODE)

    text = weird_pattern.sub(r'', text)

    text = re.sub("#", "", text)
    text = re.sub("\s+", " ", text)

    return text


def prepData(dataPath):
    xl = pd.ExcelFile(dataPath)
    data = xl.parse("Sheet1", encoding='utf-8')
    data = data.iloc[:, 0:4]
    data.columns = ['KID', 'Title', 'Problem', 'Solution']

    data = data.dropna()
    data = data.reset_index(drop=True)

    data['cleaned_Title'] = data['Title'].apply(cleaning)
    data['cleaned_Problem'] = data['Problem'].apply(cleaning)
    data['cleaned_Solution'] = data['Solution'].apply(cleaning)

    data['cleaned_Title_len_by_words'] = data['cleaned_Title'].apply(lambda t: len(hazm.word_tokenize(t)))

    data['cleaned_Title_len_by_words'] = data['cleaned_Title_len_by_words'].apply(
        lambda len_t: len_t if 1 < len_t else None)
    data = data.dropna(subset=['cleaned_Title_len_by_words'])
    data = data.reset_index(drop=True)

    data['cleaned_Problem_len_by_words'] = data['cleaned_Problem'].apply(lambda t: len(hazm.word_tokenize(t)))

    data['cleaned_Problem_len_by_words'] = data['cleaned_Problem_len_by_words'].apply(
        lambda len_t: len_t if 1 < len_t else None)
    data = data.dropna(subset=['cleaned_Problem_len_by_words'])
    data = data.reset_index(drop=True)

    data['cleaned_Solution_len_by_words'] = data['cleaned_Solution'].apply(lambda t: len(hazm.word_tokenize(t)))

    data['cleaned_Solution_len_by_words'] = data['cleaned_Solution_len_by_words'].apply(
        lambda len_t: len_t if 1 < len_t else None)
    data = data.dropna(subset=['cleaned_Solution_len_by_words'])
    data = data.reset_index(drop=True)

    data = data[['KID', 'cleaned_Title', 'cleaned_Problem', 'cleaned_Solution']]
    data.columns = ['KID', 'title', 'problem', 'solution']

    return data


class SimFinderModel():
    deprecated = False

    def __init__(self):
        self.vectors = []
        self.docs = []
        self.KIDs = []
        self.model = None
        self.didLoad = False
        self.dbClient = None
        self.vecCollection = None
        self.docCollection = None
        self.dataName = None

    def trainVectors(self, dataPath):
        data = prepData(dataPath)

        titles = data.iloc[:, 1]
        KIDs = data.iloc[:, 0]

        print('Data loaded.')

        self.model = SentenceTransformer('HooshvareLab/bert-fa-zwnj-base')
        print('Model loaded.')

        print('Calculating vectors started.')
        self.vectors = []
        self.docs = []
        for i, title in enumerate(tqdm(titles)):
            vectorDict = {}
            titleDict = {}
            sentences = sent_tokenize(title)
            embeddings_sentences = self.model.encode(sentences)
            embeddings = np.mean(np.array(embeddings_sentences), axis=0)

            vectorDict['_id'] = i
            vectorDict['vector'] = embeddings
            self.vectors.append(vectorDict)
            titleDict['_id'] = i
            titleDict['KID'] = KIDs[i]
            titleDict['doc'] = titles[i]
            self.docs.append(titleDict)

        print('Vectors calculated.')

    def load(self, forceDataReload=False):
        self.dbClient = pymongo.MongoClient(connString)
        print("SimFinderMode 3333333333")
        with open('./finderAgent/dataPath.txt') as r:
            dataPath = r.readline()

        self.dataName = dataPath.split('/')[-1].split('.')[0]
        if forceDataReload or SimFinderModel.deprecated:
            db = self.dbClient[self.dataName]
            self.vecCollection = db['vectors']
            self.docCollection = db['docs']
            print('Creating model.')

            self.trainVectors(dataPath)

            vectors = copy.deepcopy(self.vectors)
            for vec in vectors:
                vec['vector'] = Binary(pickle.dumps(vec['vector'], protocol=2))

            self.vecCollection.insert_many(vectors)
            self.docCollection.insert_many(self.docs)

            cache.set('vectors', self.vectors, timeout=None)
            cache.set('docs', self.docs, timeout=None)
            cache.set('model', self.model, timeout=None)
            print('Loaded simFinder object from data.')
            self.deprecated = False
        else:
            if len(self.vectors) == 0:
                self.vectors = cache.get('vectors')
                if self.vectors is not None:
                    self.docs = cache.get('docs')
                    self.model = cache.get('model')
                    self.KIDs = cache.get('KIDs')

                    print('Loaded simFinder object from cache.')
                    db = self.dbClient[self.dataName]
                    self.vecCollection = db['vectors']
                    self.docCollection = db['docs']
                else:
                    dbList = self.dbClient.list_database_names()
                    if self.dataName in dbList:
                        db = self.dbClient[self.dataName]
                        self.vecCollection = db['vectors']
                        self.docCollection = db['docs']

                        self.vectors = [vec for vec in self.vecCollection.find()]
                        for vec in self.vectors:
                            vec['vector'] = pickle.loads(vec['vector'])

                        self.docs = [doc for doc in self.docCollection.find()]
                        self.model = SentenceTransformer('HooshvareLab/bert-fa-zwnj-base')

                        print('Loaded simFinder object from database.')
                    else:
                        db = self.dbClient[self.dataName]
                        self.vecCollection = db['vectors']
                        self.docCollection = db['docs']
                        print('Creating model.')

                        self.trainVectors(dataPath)

                        vectors = copy.deepcopy(self.vectors)
                        for vec in vectors:
                            vec['vector'] = Binary(pickle.dumps(vec['vector'], protocol=2))

                        self.vecCollection.insert_many(vectors)
                        self.docCollection.insert_many(self.docs)
                        print('Loaded simFinder object from data.')

                    cache.set('vectors', self.vectors, timeout=None)
                    cache.set('docs', self.docs, timeout=None)
                    cache.set('model', self.model, timeout=None)

        self.didLoad = True
        print('Loading done')

    def findSimilars(self, title, numberOfHighSimilarities):
        sentences = sent_tokenize(title)
        base_embeddings_sentences = self.model.encode(sentences)
        base_embeddings = np.mean(np.array(base_embeddings_sentences), axis=0)

        rawVectors = [vec['vector'] for vec in self.vectors]
        rawDocs = [doc['doc'] for doc in self.docs]
        KIDs = [doc['KID'] for doc in self.docs]

        scores = cosine_similarity([base_embeddings], rawVectors).flatten()

        highestScoresIndexes = np.argsort(scores)[-1 * numberOfHighSimilarities:][::-1]
        highestScorers = []
        for idx in highestScoresIndexes:
            similarity = Similarity()
            similarity.doc = rawDocs[idx]
            similarity.score = scores[idx]
            similarity.KID = KIDs[idx]
            highestScorers.append(similarity)

        return highestScorers

    def updateCache(self):
        cache.set('vectors', self.vectors, timeout=None)
        cache.set('docs', self.docs, timeout=None)
        cache.set('model', self.model, timeout=None)

    def addDoc(self, doc, KID):
        vectorDict = {}
        titleDict = {}
        sentences = sent_tokenize(doc)
        embeddings_sentences = self.model.encode(sentences)
        embeddings = np.mean(np.array(embeddings_sentences), axis=0)

        currId = len(self.vectors)
        vectorDict['_id'] = currId
        vectorDict['vector'] = embeddings
        self.vectors.append(vectorDict)

        titleDict['_id'] = currId
        titleDict['doc'] = doc
        titleDict['KID'] = KID
        self.docs.append(titleDict)

        self.vecCollection.insert_one({'_id': currId, 'vector': Binary(pickle.dumps(vectorDict['vector'], protocol=2))})
        self.docCollection.insert_one(titleDict)

        self.updateCache()
