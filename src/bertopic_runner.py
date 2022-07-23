import os
import time
from collections import OrderedDict, Counter
from typing import Dict

import keras
import numpy as np
import pandas as pd
from bertopic import BERTopic
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from hdbscan import HDBSCAN
from keras.layers import Input, Dense
from keras.models import Model
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from umap import UMAP

EMBEDDING_DIR_PATH = './pretrained_models'
HUGGING_FACE_EMBEDDING_MODELS = ["all-mpnet-base-v2", "all-distilroberta-v1",
                                 "all-MiniLM-L12-v2", "all-MiniLM-L6-v2", 'paraphrase-multilingual-MiniLM-L12-v2']


def download_embedding_models(embedding_folder: str) -> None:
    if not os.path.exists(embedding_folder):
        print(f'[INFO] The embedding folder "{embedding_folder}" download folder was missing, so creating..')
        os.mkdir(embedding_folder)

    for embedding_model in HUGGING_FACE_EMBEDDING_MODELS:
        target_path = f'{embedding_folder}/sentence-transformers_{embedding_model}'
        if not os.path.exists(target_path):
            print(f'[INFO] The embedding model folder:"{target_path}" not found, downloading..')
            SentenceTransformer(model_name_or_path=embedding_model, cache_folder=embedding_folder)
            print(f'[INFO] The embedding model folder:"{target_path}" downloaded.')
        else:
            print(f'[INFO] The embedding model folder:"{target_path}" found, so no need to download.')


class BertopicTrainer:
    def __init__(
            self,
            dataset: str,
            model_name: str,
            args: Dict,
            run_id: int,
            verbose: bool = True
    ):
        assert args["embedding_model"] in HUGGING_FACE_EMBEDDING_MODELS, \
            f'"{args["embedding_model"]}" must be in {HUGGING_FACE_EMBEDDING_MODELS}!'
        download_embedding_models(embedding_folder=EMBEDDING_DIR_PATH)

        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        if model_name not in {'bertopic', 'lda-bert'}:
            raise Exception(f'Model name {model_name} is not in [bertopic, lda-bert]!')

        print('[INFO] Embeddings are being encoded..')
        if args["embedding_model"] != 'doc2vec':
            emb_model = SentenceTransformer(f'{EMBEDDING_DIR_PATH}/sentence-transformers_{args["embedding_model"]}')
            self.embeddings = emb_model.encode(args['docs'], show_progress_bar=True)
        else:
            self.embeddings = None
        print('[INFO] Embeddings encoded successfully.')

        print('[INFO] Embedding Model created with SentenceTransformer.')
        self.dataset = dataset
        self.model_name = model_name

        self.verbose = verbose
        self.docs = args['docs']
        self.labels = args['labels']
        self.params = args
        self.run_id = run_id

        method_specific_params = args.copy()
        for key in ['docs', 'labels', 'algorithm', 'data_name']:
            method_specific_params.pop(key, None)
        self.method_specific_params = method_specific_params

    # TODO: Maybe save param?
    def train(self):
        topic_model, df_output_doc_topic, df_output_topic_word = self._train_tm_model(params=self.params)
        return topic_model, df_output_doc_topic, df_output_topic_word

    def _train_tm_model(self, params):

        if self.model_name == "bertopic":
            return self._train_bertopic_model(params)
        elif self.model_name == "lda-bert":
            return self._train_lda_bert_model(params)

    def _train_bertopic_model(self, params):
        # TODO: Other cluster method with assignment score
        if params["cluster_model"] == "hdbscan":
            cluster_model = HDBSCAN(**params["hdbscan_args"])
        elif params["cluster_model"] == "kmeans":
            cluster_model = KMeans(n_clusters=params["num_topics"])
        else:
            raise ValueError(f'{params["cluster_model"]} is not recognized, should be "hdbscan" or "kmeans".')

        topic_model = BERTopic(
            # embedding_model=params["embedding_model"],
            verbose=self.verbose,
            top_n_words=params["top_n_words"],
            n_gram_range=params["n_gram_range_tuple"],
            min_topic_size=params["min_docs_per_topic"],
            umap_model=UMAP(**params["umap_args"]),
            hdbscan_model=cluster_model
            # nr_topics = params["num_topics"]
        )
        # Train and fit model
        t0 = time.time()
        topics, probs = topic_model.fit_transform(self.docs, embeddings=self.embeddings)

        num_detected_topics = len(set(topics))

        # Reduce nr_topics if needed
        if (params['num_topics'] is not None) and (params['num_topics'] < num_detected_topics):
            self.reduced = True
            topics, probs = topic_model.reduce_topics(self.docs, topics, probs, nr_topics=params['num_topics'])
            # print(f'The shape of probs after reducing is: {len(probs)} x {len(probs[0])}')
            # probs = [prob[0:params['num_topics']] for prob in probs]
            # print(f'The shape of probs after correction is: {len(probs)} x {len(probs[0])}')
        else:
            self.reduced = False

        num_final_topics = len(set(topics))

        duration = time.time() - t0

        # Evaluate metric(s)
        topic_list = []
        word_score_list = []

        # Iterate over topics to create nested list of topics
        for i in topic_model.get_topic_info()['Topic']:
            single_topic_list = []
            single_word_score_list = []
            for elem in topic_model.get_topic(i):
                single_topic_list.append(elem[0])
                single_word_score_list.append(elem[1])

            topic_list.append(single_topic_list)
            word_score_list.append(single_word_score_list)

        # Model_output as dictionary with key "topics"
        model_output = {"topics": topic_list, 'word_scores': word_score_list}

        # Get representative docs
        # rep_docs = []
        # for i in topic_model.get_topic_info()['Topic']:
        #    rep_docs.append(topic_model.get_representative_docs(i)[:10])

        # Construct df doc_topic
        doc_topic_dict = {
            "run_id": self.run_id,
            "Document ID": range(len(self.docs)),
            "Document": self.docs,
            "Real Label": self.labels,
            "Assigned Topic Num": topics if (params["hdbscan_args"] is not None) else [t - 1 for t in topics],
            "Assignment Score": probs if (params["hdbscan_args"] is not None) else 1
        }

        df_output_doc_topic = pd.DataFrame(doc_topic_dict)

        # Construct df topic_word
        nrows_topic_word = len(topic_model.get_topic_info()['Topic'])

        params_dict = OrderedDict([
            ('run_id', self.run_id),
            ('method', self.model_name),
            ('method_specific_params', self.method_specific_params),
            ('dataset', self.dataset),
            ('num_given_topics', params['num_topics']),
            ('reduced', self.reduced),
        ])

        topic_word_dict = {
            'topic_num': topic_model.get_topic_info()['Topic'],
            'topic_size': topic_model.get_topic_info()['Count'],
            'topic_words': model_output['topics'],
            'word_scores': model_output['word_scores']
        }

        results_dict = OrderedDict(
            [('num_detected_topics', num_detected_topics), ('num_final_topics', num_final_topics),
             ('duration_secs', duration)])

        params_df = pd.DataFrame([params_dict] * nrows_topic_word)
        results_df = pd.DataFrame([results_dict] * nrows_topic_word)

        df_output_topic_word = pd.concat([params_df, pd.DataFrame(topic_word_dict), results_df], axis=1)

        return topic_model, df_output_doc_topic, df_output_topic_word

    def _train_lda_bert_model(self, params):

        t0 = time.time()

        lda_bert_params = {
            'num_topics': params['num_topics'],
            'top_n_words': params['top_n_words'],
            'embedding_model': params['embedding_model'],
            'gamma': params['gamma'],
            'random_state': params['random_state']
        }

        topic_model = LdaBert(self.embeddings, lda_bert_params)
        pred_topic_labels, topic_words, word_scores = topic_model.fit_transform(self.docs)
        t1 = time.time()
        duration = t1 - t0

        # Construct df doc_topic
        doc_topic_dict = {
            "run_id": self.run_id,
            "Document ID": range(len(self.docs)),
            "Document": self.docs,
            "Real Label": self.labels,
            "Assigned Topic Num": pred_topic_labels,
            "Assignment Score": int(1)
        }

        df_output_doc_topic = pd.DataFrame(doc_topic_dict)

        nrows_topic_word = params['num_topics']

        params_dict = OrderedDict([
            ('run_id', self.run_id),
            ('method', self.model_name),
            ('method_specific_params', self.method_specific_params),
            ('dataset', self.dataset),
            ('num_given_topics', params['num_topics']),
            ('reduced', False),
        ])

        topic_word_dict = {
            'topic_num': range(nrows_topic_word),
            'topic_size': np.unique(pred_topic_labels, return_counts=True)[1],
            'topic_words': topic_words,
            'word_scores': word_scores  # TODO: use c-Tf-IDF for topic_words and word_scores
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

    # def evaluate(self, output_tm):
    #
    #     results = {}
    #
    #     for scorers, _ in self.metrics:
    #         for scorer, name in scorers:
    #             score = scorer.score(output_tm)
    #             results[name] = float(score)
    #
    #     if self.verbose:
    #         print(">>>  Results")
    #         for metric, score in results.items():
    #             print(f"The topic model {self.model_name} has {metric} score: {score}")
    #         print("")
    #
    #     return results

    # def get_metrics(self):
    #
    #     if isinstance(self.data[0], list):
    #         text = self.data
    #     else:
    #         text = [d.split(" ") for d in self.data]
    #
    #     topic_coherence = Coherence(texts=text, topk=self.top_k, measure="c_v")
    #     topic_diversity = TopicDiversity(topk=self.top_k)
    #
    #    # Define methods
    #     coherence = [(topic_coherence, "c_v")]
    #     diversity = [(topic_diversity, "diversity")]
    #     metrics = [(coherence, "Coherence"), (diversity, "Diversity")]
    #
    #     return metrics

    # def get_dataset(self):
    #     data = load_documents(self.dataset)
    #     return data


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
        # decoder_layer = self.autoencoder.layers[-1]
        self.decoder = Model(encoded_input, self.autoencoder.layers[-1](encoded_input))
        self.autoencoder.compile(optimizer='adam', loss=keras.losses.mean_squared_error)

    def fit(self, x_data):
        if not self.autoencoder:
            self._compile(x_data.shape[1])
        x_train, x_test = train_test_split(x_data)
        self.his = self.autoencoder.fit(
            x_train, x_train,
            epochs=200,
            batch_size=128,
            shuffle=True,
            validation_data=(x_test, x_test), verbose=0)


class LdaBert:

    def __init__(self, embeddings, params):

        self.params = params
        self.model_name = 'lda-bert'
        # ## LDA-Bert specific params ####

        self.vec = {}  # Init empty dict for LDA-BERT

        self.embeddings = embeddings  # Init pretrained embeddings

        # self.k = k -> here: params['num_topics']
        self.dictionary = None
        self.corpus = None

        # Init Kmeans here with nr of topics
        self.cluster_model = KMeans(n_clusters=self.params['num_topics'])
        self.lda_model = None

        # self.gamma = 15  -> here: params['gamma'] # parameter for relative importance of lda
        self.AE = None

        # ### END: LDA-Bert specific params ####

    def vectorize(self, sentences, token_lists, method):
        """
        Get vector representations for selected methods: LDA, BERT and LDA-BERT
        """

        if method == 'LDA':
            print('[INFO] Getting vector representations for LDA ...')
            if not self.lda_model:
                self.lda_model = LdaModel(self.corpus, num_topics=self.params['num_topics'], id2word=self.dictionary,
                                          passes=20, random_state=self.params['random_state'])

            def get_vec_lda(a_model, corpus, k):
                """
                Get the LDA vector representation (probabilistic topic assignments for all documents)
                :return: lda_vectors with dimension: (n_doc * n_topic)
                """
                n_doc = len(corpus)
                lda_vectors = np.zeros((n_doc, k))

                for i in range(n_doc):
                    # get the distribution for the i-th document in corpus
                    for topic, prob in a_model.get_document_topics(corpus[i]):
                        lda_vectors[i, topic] = prob

                return lda_vectors

            vec = get_vec_lda(self.lda_model, self.corpus, self.params['num_topics'])
            print('Getting vector representations for LDA. Done!')

            return vec

        elif method == 'BERT':
            print('[INFO] Getting vector representations for BERT ...')
            if self.embeddings is not None:
                vec = np.array(self.embeddings)
            else:
                model = SentenceTransformer(self.params['embedding_model'])
                vec = np.array(model.encode(sentences, show_progress_bar=True))

            print('[INFO] Getting vector representations for BERT. Done!')
            return vec

        elif method == 'lda-bert':

            vec_lda = self.vectorize(sentences, token_lists, method='LDA')
            vec_bert = self.vectorize(sentences, token_lists, method='BERT')

            # Concat lda and bert vectors with HP gamma
            vec_lda_bert = np.c_[vec_lda * self.params['gamma'], vec_bert]

            self.vec['LDA'] = vec_lda
            self.vec['BERT'] = vec_bert
            self.vec['LDA_BERT_FULL'] = vec_lda_bert

            if not self.AE:
                self.AE = Autoencoder()
                print('Fitting Autoencoder ...')
                self.AE.fit(vec_lda_bert)
                print('Fitting Autoencoder Done!')

            vec = self.AE.encoder.predict(vec_lda_bert)
            return vec

    @staticmethod
    def get_topic_words(token_lists, labels, k=None, top_n=10):
        """
        get most frequent words within each topic from clustering results as topic words
        """
        if k is None:
            k = len(np.unique(labels))

        # raise Exception("Debug here")
        topics = ['' for _ in range(k)]

        for i, c in enumerate(token_lists):
            topics[labels[i]] += (' ' + ' '.join(c))

        word_counts = list(map(lambda x: Counter(x.split()).items(), topics))
        total_sum_words = list(map(lambda x: len(x.split()), topics))

        # get sorted word counts
        word_counts = list(map(lambda x: sorted(x, key=lambda y: y[1], reverse=True), word_counts))

        # get topics
        topic_words = list(map(lambda x: list(map(lambda y: y[0], x[:top_n])), word_counts))
        top_word_counts = list(map(lambda x: list(map(lambda y: y[1], x[:top_n])), word_counts))

        for i in range(k):
            for n, number in enumerate(top_word_counts[i]):
                top_word_counts[i][n] = float(number / total_sum_words[i])

        return topic_words, top_word_counts

    # @staticmethod
    # def preprocess_sent(raw):
    #     """
    #     returns raw sentences without preprocessing (should be done in preprocessing module)
    #     """
    #     return raw

    # @staticmethod
    # def preprocess_word(s):
    #     """
    #     returns sentence as list of tokens (word list)
    #     """
    #     word_list = word_tokenize(s)
    #
    #     return word_list

    @staticmethod
    def preprocess(docs):
        """
        Preprocess the data by calling preprocess_sent and word
        returns sentences (List[str]) and token_lists (List[List[str]])
        """
        print('[INFO] Tokenizing raw texts...')

        sentences = docs  # sentence level preprocessed

        token_lists = [word_tokenize(s) for s in sentences]  # word level preprocessed
        print('[INFO] Tokenizing raw texts. Done!')

        return sentences, token_lists

    def fit_transform(self, docs):
        sentences, token_lists = self.preprocess(docs)

        # turn tokenized documents into an id <-> term dictionary
        if not self.dictionary:
            self.dictionary = corpora.Dictionary(token_lists)
            # convert tokenized documents into a document-term matrix
            self.corpus = [self.dictionary.doc2bow(text) for text in token_lists]

        #  Getting vector representations ####
        print('Clustering embeddings ...')

        # Init cluster_model above
        # self.cluster_model = m_clustering(self.k)
        self.vec[self.model_name] = self.vectorize(sentences, token_lists, self.model_name)
        self.cluster_model.fit(self.vec[self.model_name])

        print('Clustering embeddings. Done!')

        pred_topic_labels = self.cluster_model.labels_

        # TODO: Call correct function for c-TF-IDF
        topic_words, word_scores = self.get_topic_words(token_lists, pred_topic_labels, k=self.params['num_topics'],
                                                        top_n=self.params['top_n_words'])

        return pred_topic_labels, topic_words, word_scores


def test_lda_bert():
    # from src.utils import load_documents
    # data, labels = load_documents('crisis_toy')
    # Encode data with embedding model
    # emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    # embeddings = emb_model.encode(data, show_progress_bar=True)

    params = {
        'embedding_model': "all-MiniLM-L6-v2",
        'num_topics': 2,
        'top_n_words': 10,
        'gamma': 15
    }

    trainer = BertopicTrainer(dataset='crisis_toy',
                              model_name="lda-bert",
                              args=params,
                              run_id=12345,
                              # custom_model=None,
                              verbose=True,
                              )

    lda_bert_model, df_output_doc_topic, df_output_topic_word = trainer.train()
    print(lda_bert_model, df_output_doc_topic, df_output_topic_word)


if __name__ == "__main__":
    test_lda_bert()
