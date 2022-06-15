from src.utils import load_documents

from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.coherence_metrics import Coherence

from bertopic import BERTopic

import time

import numpy as np

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

        # Prepare data and metrics
        self.data = self.get_dataset()
        self.metrics = self.get_metrics()

        # CTM
        self.qt_ctm = None
        self.training_dataset_ctm = None

    ## Maybe save param?
    def train(self):

        output, duration = self._train_tm_model(params = self.params)
        scores = self.evaluate(output)

        result = {
                "Dataset": self.dataset,
                "Dataset Size": len(self.data),
                "Model": self.model_name,
                "Params": self.params,
                "Scores": scores,
                "Computation Time": duration,
            }

        return result

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
                               nr_topics = params["number_topics"])
    
        ## Train and fit model
        t0 = time.time()
        topics, probs = topic_model.fit_transform(self.data, self.embeddings)
        t1 = time.time()
    
        duration = t1 - t0

        ## Evaluate metric(s)
        topics = []
    
        ## Iterate over topics to create nested list of topics
        for i in range(0, len(topic_model.get_topics()) - 1):
            single_topic_list = []
            for j in range(len(topic_model.get_topic(i))):
                single_topic_list.append(topic_model.get_topic(i)[j][0])
        
            topics.append(single_topic_list)

    

        ## Model_output as dictionary with key "topics"
        model_output = {}

        model_output["topics"] = topics

        return model_output, duration

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

        topic_coherence = Coherence(texts=[d.split(" ") for d in self.data], topk=self.topk, measure="c_v")
        topic_diversity = TopicDiversity(topk=self.topk)

        # Define methods
        coherence = [(topic_coherence, "c_v")]
        diversity = [(topic_diversity, "diversity")]
        metrics = [(coherence, "Coherence"), (diversity, "Diversity")]

        return metrics
    
    def get_dataset(self):

        if self.dataset == "crisis_12":
            dir = './data/crisis_resource_12_labeled_by_paid_workers'
            col = 'text'
        elif self.dataset == "crisis_1":
            dir = './data/crisis_resource_1_labeled_by_paid_workers'
            col = 'text'
        elif self.dataset == "20news":
            dir = './data/20news_bydate'
            col = 'text'

        data = load_documents(dataset_dir = dir, dataset_text_col = 'text')

        return data
