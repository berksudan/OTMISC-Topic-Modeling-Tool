from typing import OrderedDict
#from utils import load_documents
from src.utils import load_documents
from collections import OrderedDict, Counter

from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.coherence_metrics import Coherence

from bertopic import BERTopic

import time

import numpy as np

import pandas as pd

#import umap
from umap import UMAP
from hdbscan import HDBSCAN

#import os
#import json

from tqdm import tqdm

from nltk.corpus import wordnet
import re
import matplotlib.pyplot as plt
from nltk.corpus import stopwords 

import gensim
from gensim.utils import simple_preprocess
from gensim import corpora, models
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *

import datetime
import time
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D

import nltk
from nltk.tokenize import word_tokenize

import keras
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.model_selection import train_test_split

from sklearn.cluster import KMeans
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from sentence_transformers import SentenceTransformer

class Trainer:

    def __init__(
        self,
        dataset: str,
        model_name: str,
        params: None,
        topk: int = 10,
        custom_dataset: bool = False,
        bt_embeddings: np.ndarray = None,
        custom_model=None,
        verbose: bool = True
    ):
        self.dataset = dataset
        self.custom_dataset = custom_dataset
        self.model_name = model_name

        if self.model_name not in {'bertopic', 'lda-bert'}:
            raise Exception(f'Model name {self.model_name} is not in [bertopic, lda-bert]!')

        self.params = params
        self.topk = topk
        
        self.embeddings = bt_embeddings
        #self.ctm_preprocessed_docs = None
        
        self.custom_model = custom_model
        self.verbose = verbose

        # Prepare data and metrics
        self.data = self.get_dataset()
        self.docs = self.data[0]
        self.labels = self.data[1]
        self.metrics = self.get_metrics()

        # CTM
        self.qt_ctm = None
        self.training_dataset_ctm = None

    ## Maybe save param?
    def train(self):

        #output, duration, df, plot, rep_docs, topics, probs, words_score, df_output_doc_topic = self._train_tm_model(params = self.params)
        
        #output, topic_model, df_output_doc_topic, df_output_topic_word = self._train_tm_model(params = self.params)
        topic_model, df_output_doc_topic, df_output_topic_word = self._train_tm_model(params = self.params)
        
        #pred_labels, topic_words = self._train_tm_model(params = self.params)
        #scores = self.evaluate(output)

        #result = {
        #        "Dataset": self.dataset,
        #        "Dataset Size": len(self.data),
        #        "Model": self.model_name,
        #        "Params": self.params,
        #        "Scores": scores,
        #        "Computation Time": duration,
        #        "Topic Info": df,
        #        "Barchart": plot,
        #        "Rep Docs": rep_docs,
        #        "Topics": topics,
        #        "Probs": probs,
        #        "Words score": words_score
        #    }

        return topic_model, df_output_doc_topic, df_output_topic_word
        #return pred_labels, topic_words

    def _train_tm_model(self, params):

        if self.model_name == "bertopic":
            return self._train_bertopic_model(params)
        elif self.model_name == "lda-bert":
            return self._train_lda_bert_model(params)

    def _train_bertopic_model(self, params):
        ## Define BERTopic model
        topic_model = BERTopic(embedding_model = params["embedding_model"], 
                               verbose = self.verbose,
                               top_n_words = params["top_n_words"],
                               n_gram_range = params["n_gram_range_tuple"],
                               min_topic_size = params["min_docs_per_topic"],
                               umap_model= UMAP(**params["umap_args"]),
                               hdbscan_model = HDBSCAN(**params["hdbscan_args"])
                               #nr_topics = params["number_topics"]
                               )
    
        ## Train and fit model
        t0 = time.time()
        topics, probs = topic_model.fit_transform(self.docs, self.embeddings)

        num_detected_topics = len(set(topics))

        ## Reduce nr_topics if needed
        if (params['number_topics'] is not None) and (params['number_topics'] < num_detected_topics):
            self.reduced = True
            topics, probs = topic_model.reduce_topics(self.docs, topics, probs, nr_topics=params['number_topics'])
        else:
            self.reduced = False
        
        num_final_topics = len(set(topics))

        t1 = time.time()
    
        duration = t1 - t0

        #print(topics)
        #print(probs)
        ## Evaluate metric(s)
        topic_list = []
        word_score_list = []
    
        ## Iterate over topics to create nested list of topics
        for i in topic_model.get_topic_info()['Topic']:
            single_topic_list = []
            single_word_score_list = []
            for elem in topic_model.get_topic(i):
                single_topic_list.append(elem[0])
                single_word_score_list.append(elem[1])
        
            topic_list.append(single_topic_list)
            word_score_list.append(single_word_score_list)

    

        ## Model_output as dictionary with key "topics"
        model_output = {}

        model_output["topics"] = topic_list
        model_output['word_scores'] = word_score_list

        ## Get representative docs
        rep_docs = []
        for i in topic_model.get_topic_info()['Topic']:
            rep_docs.append(topic_model.get_representative_docs(i)[:10])

        ## Construct df doc_topic
        doc_topic_dict = {
            "run_id": int(t0),
            "Document ID": range(len(self.docs)),
            "Document": self.docs,
            "Real Label": self.labels,
            "Assigned Topic Num": topics,
            "Assignment Score": probs
        }

        df_output_doc_topic = pd.DataFrame(doc_topic_dict)

        ## Construct df topic_word
        nrows_topic_word = len(topic_model.get_topic_info()['Topic'])

        params_dict = OrderedDict([
            ('run_id', int(t0)),
            ('method', self.model_name),
            ('method_specific_params', params),
            ('dataset', self.dataset),
            ('num_given_topics', params['number_topics']),
            ('reduced', self.reduced),
        ])

        topic_word_dict = {
            'topic_num': topic_model.get_topic_info()['Topic'],
            'topic_size': topic_model.get_topic_info()['Count'],
            'topic_words': model_output['topics'],
            'word_scores': model_output['word_scores']
        }
        
        results_dict = OrderedDict([
            ('num_detected_topics', num_detected_topics),
            ('num_final_topics', num_final_topics),
            ('duration_secs', duration),
        ])

        params_df = pd.DataFrame([params_dict] * nrows_topic_word)
        results_df = pd.DataFrame([results_dict] * nrows_topic_word)

        df_output_topic_word = pd.concat([params_df, pd.DataFrame(topic_word_dict), results_df], axis=1)
        
        return topic_model, df_output_doc_topic, df_output_topic_word
        #return model_output, topic_model, df_output_doc_topic, df_output_topic_word
        #return model_output, duration, topic_model.get_topic_info(), topic_model.visualize_barchart(), rep_docs, topics, probs, topic_model.get_topics(), df_output_doc_topic

    def _train_lda_bert_model(self, params):

        t0 = time.time()

        lda_bert_params = {
            'number_topics': params['number_topics'], 
            'top_n_words': params['top_n_words'],
            'embedding_model': params['embedding_model'],
            'gamma': params['gamma']
        }

        topic_model = LDABERT(self.embeddings, lda_bert_params)

        pred_topic_labels, topic_words, word_scores = topic_model.fit_transform(self.docs)

        t1 = time.time()

        duration = t1 - t0

        ## Construct df doc_topic
        doc_topic_dict = {
            "run_id": int(t0),
            "Document ID": range(len(self.docs)),
            "Document": self.docs,
            "Real Label": self.labels,
            "Assigned Topic Num": pred_topic_labels,
            "Assignment Score": int(1)
        }

        df_output_doc_topic = pd.DataFrame(doc_topic_dict)

        ## Construct df topic_word
        print(f'the words scores are: {word_scores}')
        print(f'and have length: {len(word_scores)}')

        nrows_topic_word = params['number_topics']

        params_dict = OrderedDict([
            ('run_id', int(t0)),
            ('method', self.model_name),
            ('method_specific_params', params),
            ('dataset', self.dataset),
            ('num_given_topics', params['number_topics']),
            ('reduced', False),
        ])

        topic_word_dict = {
            'topic_num': range(nrows_topic_word),
            'topic_size': np.unique(pred_topic_labels, return_counts=True)[1],
            'topic_words': topic_words,
            'word_scores': word_scores ##TODO: use frequency as word score OR use c-Tf-IDF for topic_words and word_scores
        }
        
        results_dict = OrderedDict([
            ('num_detected_topics', nrows_topic_word),
            ('num_final_topics', nrows_topic_word),
            ('duration_secs', duration),
        ])

        params_df = pd.DataFrame([params_dict] * nrows_topic_word)
        results_df = pd.DataFrame([results_dict] * nrows_topic_word)

        df_output_topic_word = pd.concat([params_df, pd.DataFrame(topic_word_dict), results_df], axis=1)
        
        return topic_model, df_output_doc_topic, df_output_topic_word

    def evaluate(self, output_tm):

        results = {}

        for scorers, _ in self.metrics:
            for scorer, name in scorers:
                score = scorer.score(output_tm)
                results[name] = float(score)
            
        if self.verbose:
            print(">>>  Results")
            for metric, score in results.items():
                print(f"The topic model {self.model_name} has {metric} score: {score}")
            print("")
        
        return results

    def get_metrics(self):
        
        if isinstance(self.data[0], list):
            text = self.data
        else:
            text = [d.split(" ") for d in self.data]
        
        topic_coherence = Coherence(texts=text, topk=self.topk, measure="c_v")
        topic_diversity = TopicDiversity(topk=self.topk)

        # Define methods
        coherence = [(topic_coherence, "c_v")]
        diversity = [(topic_diversity, "diversity")]
        metrics = [(coherence, "Coherence"), (diversity, "Diversity")]

        return metrics
    
    def get_dataset(self):

        data = load_documents(self.dataset)

        return data

class Autoencoder:
    """
    Autoencoder for learning latent space representation
    architecture simplified for only one hidden layer
    """

    def __init__(self, latent_dim=32, activation='relu', epochs=200, batch_size=128):
        self.latent_dim = latent_dim
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.his = None

    def _compile(self, input_dim):
        """
        compile the computational graph
        """
        input_vec = Input(shape=(input_dim,))
        encoded = Dense(self.latent_dim, activation=self.activation)(input_vec)
        decoded = Dense(input_dim, activation=self.activation)(encoded)
        self.autoencoder = Model(input_vec, decoded)
        self.encoder = Model(input_vec, encoded)
        encoded_input = Input(shape=(self.latent_dim,))
        decoder_layer = self.autoencoder.layers[-1]
        self.decoder = Model(encoded_input, self.autoencoder.layers[-1](encoded_input))
        self.autoencoder.compile(optimizer='adam', loss=keras.losses.mean_squared_error)

    def fit(self, X):
        if not self.autoencoder:
            self._compile(X.shape[1])
        X_train, X_test = train_test_split(X)
        self.his = self.autoencoder.fit(X_train, X_train,
                                        epochs=200,
                                        batch_size=128,
                                        shuffle=True,
                                        validation_data=(X_test, X_test), verbose=0)

class LDABERT:

    def __init__(self, embeddings, params):
        
        self.params = params
        self.model_name = 'lda-bert'
        #### LDA-Bert specific params ####
        
        # Init empty dict for LDA-BERT
        self.vec = {}

        # Init pretrained embeddings
        self.embeddings = embeddings

        
        #self.k = k -> here: params['number_topics']
        self.dictionary = None
        self.corpus = None
        
        # TODO: Init Kmeans here with nr of topics
        self.cluster_model = KMeans(n_clusters=self.params['number_topics'])
        self.ldamodel = None
        # parameter for reletive importance of lda
        #self.gamma = 15  -> here: params['gamma']
        #self.method = method -> here: self.model_name
        self.AE = None
        
        #### END: LDA-Bert specific params ####

    def vectorize(self, sentences, token_lists, method):
        """
        Get vector representations for selected methods: LDA, BERT and LDA-BERT
        """

        if method == 'LDA':

            print('Getting vector representations for LDA ...')
            if not self.ldamodel:
                self.ldamodel = LdaModel(self.corpus, num_topics=self.params['number_topics'], id2word=self.dictionary, passes=20)

            def get_vec_lda(model, corpus, k):
                """
                Get the LDA vector representation (probabilistic topic assignments for all documents)
                :return: vec_lda with dimension: (n_doc * n_topic)
                """
                n_doc = len(corpus)
                vec_lda = np.zeros((n_doc, k))

                for i in range(n_doc):
                    # get the distribution for the i-th document in corpus
                    for topic, prob in model.get_document_topics(corpus[i]):
                        vec_lda[i, topic] = prob

                return vec_lda

            vec = get_vec_lda(self.ldamodel, self.corpus, self.params['number_topics'])
            print('Getting vector representations for LDA. Done!')

            return vec

        elif method == 'BERT':

            print('Getting vector representations for BERT ...')

            if self.embeddings is not None:
                vec = np.array(self.embeddings)
            else:
                model = SentenceTransformer(self.params['embedding_model'])
                vec = np.array(model.encode(sentences, show_progress_bar=True))
            
            print('Getting vector representations for BERT. Done!')

            return vec

        elif method == 'lda-bert':

            vec_lda = self.vectorize(sentences, token_lists, method='LDA')
            vec_bert = self.vectorize(sentences, token_lists, method='BERT')

            # Concat lda and bert vectors with HP gamma
            vec_ldabert = np.c_[vec_lda * self.params['gamma'], vec_bert]

            self.vec['LDA'] = vec_lda
            self.vec['BERT'] = vec_bert
            self.vec['LDA_BERT_FULL'] = vec_ldabert

            if not self.AE:
                self.AE = Autoencoder()
                print('Fitting Autoencoder ...')
                self.AE.fit(vec_ldabert)
                print('Fitting Autoencoder Done!')

            vec = self.AE.encoder.predict(vec_ldabert)
            return vec

    def get_topic_words(self, token_lists, labels, k=None, top_n = 10):
            """
            get most frequent words within each topic from clustering results as topic words
            """
            if k is None:
                k = len(np.unique(labels))

            #raise Exception("Debug here")
            topics = ['' for _ in range(k)]

            for i, c in enumerate(token_lists):
                topics[labels[i]] += (' ' + ' '.join(c))

            word_counts = list(map(lambda x: Counter(x.split()).items(), topics))
            total_sum_words = list(map(lambda x: len(x.split()), topics))

            # get sorted word counts
            word_counts = list(map(lambda x: sorted(x, key=lambda x: x[1], reverse=True), word_counts))

            # get topics
            topic_words = list(map(lambda x: list(map(lambda x: x[0], x[:top_n])), word_counts))
            top_word_counts = list(map(lambda x: list(map(lambda x: x[1], x[:top_n])), word_counts))
            
            for i in range(k):
                for n, number in enumerate(top_word_counts[i]):
                    top_word_counts[i][n] = float(number / total_sum_words[i])

            return topic_words, top_word_counts
    
    @staticmethod
    def preprocess_sent(raw):
            """
            returns raw sentences without preprocessing (should be done in preprocessing module)
            """
            return raw

    @staticmethod
    def preprocess_word(s):
        """
        returns sentence as list of tokens (word list)
        """
        word_list = word_tokenize(s)

        return word_list

    def preprocess(self, docs):
        """
        Preprocess the data by calling preprocess_sent and word
        returns sentences (List[str]) and token_lists (List[List[str]])
        """
        print('Preprocessing raw texts ...')

        n_docs = len(docs)

        # sentence level preprocessed
        sentences = []

        # word level preprocessed
        token_lists = []  
        
        for i, doc in enumerate(docs):
            sentence = self.preprocess_sent(doc)
            token_list = self.preprocess_word(sentence)
            if token_list:
                sentences.append(sentence)
                token_lists.append(token_list)

            print('{} %'.format(str(np.round((i + 1) / n_docs * 100, 2))), end='\r')

        print('Preprocessing raw texts. Done!')
        
        return sentences, token_lists
    
    def fit_transform(self, docs):
        sentences, token_lists = self.preprocess(docs)

        # turn tokenized documents into a id <-> term dictionary
        if not self.dictionary:
            self.dictionary = corpora.Dictionary(token_lists)
            # convert tokenized documents into a document-term matrix
            self.corpus = [self.dictionary.doc2bow(text) for text in token_lists]

        #### Getting vector representations ####
        print('Clustering embeddings ...')

        # Init cluster_model above
        # self.cluster_model = m_clustering(self.k)
        self.vec[self.model_name] = self.vectorize(sentences, token_lists, self.model_name)
        self.cluster_model.fit(self.vec[self.model_name])

        print('Clustering embeddings. Done!')

        pred_topic_labels = self.cluster_model.labels_

        topic_words, word_scores = self.get_topic_words(token_lists, pred_topic_labels, k = self.params['number_topics'], top_n = self.params['top_n_words'])

        return pred_topic_labels, topic_words, word_scores

def test_lda_bert():
    
    data, labels = load_documents('crisis_toy')
        
    # Encode data with embedding model
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = emb_model.encode(data, show_progress_bar=True)
    
    
    params = {
        'embedding_model': "all-MiniLM-L6-v2",
        'number_topics': 2,
        'top_n_words': 10,
        'gamma': 15 
    }
    
    
    trainer = Trainer(dataset = 'crisis_toy',
                      model_name = "lda-bert",
                      params = params,
                      topk = 10,
                      bt_embeddings = embeddings,
                      custom_model = None,
                      verbose = True,
                      )
    
    model, df_output_doc_topic, df_output_topic_word = trainer.train()

if __name__ == "__main__":
    test_lda_bert()
