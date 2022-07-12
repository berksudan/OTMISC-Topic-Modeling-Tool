from typing import OrderedDict
from src.utils import load_documents
from collections import OrderedDict

from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.coherence_metrics import Coherence

from bertopic import BERTopic

import time

import numpy as np

import pandas as pd

from umap import UMAP
from hdbscan import HDBSCAN

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
        self.params = params
        self.topk = topk
        self.embeddings = bt_embeddings
        self.ctm_preprocessed_docs = None
        self.custom_model = custom_model
        self.verbose = verbose
        #self.reduced = True if params["number_topics"] is not None else False

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
        output, topic_model, df_output_doc_topic, df_output_topic_word = self._train_tm_model(params = self.params)
        #output, topic_model, df_output_doc_topic = self._train_tm_model(params = self.params)
        scores = self.evaluate(output)

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

    def _train_tm_model(self, params):

        if self.model_name == "bertopic":
            return self._train_bertopic_model(params)

    def _train_bertopic_model(self, params):
        ## Define BERTopic model
        topic_model = BERTopic(embedding_model = params["embedding_model"], 
                               verbose = self.verbose,
                               top_n_words = params["topic_n_words"],
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
        
        return model_output, topic_model, df_output_doc_topic, df_output_topic_word
        #return model_output, duration, topic_model.get_topic_info(), topic_model.visualize_barchart(), rep_docs, topics, probs, topic_model.get_topics(), df_output_doc_topic

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

        #if self.dataset == "crisis_12":
        #    dir_str = './data/crisis_resource_12_labeled_by_paid_workers'
        #    col = 'text'
        #elif self.dataset == "crisis_12_preprocessed":
        #    dir_str = './data/crisis_resource_12_labeled_by_paid_workers_preprocessed'
        #    col = 'Tweets'
        #elif self.dataset == "crisis_1":
        #    dir_str = './data/crisis_resource_01_labeled_by_paid_workers'
        #    col = 'tweet_text'
        #elif self.dataset == "crisis_1_preprocessed":
        #    dir_str = './data/crisis_resource_01_labeled_by_paid_workers_preprocessed'
        #    col = 'Tweets'
        #elif self.dataset == "20news":
        #    dir_str = './data/20news_bydate'
        #    col = 'text'

        data = load_documents(self.dataset)

        #if self.dataset != "20news":
        #    data = data[0]

        return data


if __name__ == "__main__":
    print("Hello")
