import pandas as pd
import unicodedata
import re
import contractions
from pathlib import Path
import numpy as np
import time

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import octis
from octis.models.LDA import LDA
from octis.models.NMF import NMF
from octis.dataset.dataset import Dataset
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.coherence_metrics import Coherence

from src import visualizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


# Please choose the entries from the following sets:
    
# model: ['LDA', 'NMF']
# dataset: ['crisis_12', '20news']

# Other parameters
# top_n_topics: int (This applies only when vis_labels_per_topic is True, it decides how many topics are shown)
# n_words: int ((This applies only when vis_barchart is True, it decides how many words are shown))

# Default setting, you can set your own parameters in the notebook
settings_example = {'model': 'LDA',
                    'dataset': 'crisis_12',
                    'top_n_topics': 5,
                    'n_words': 5}

valid_models = ['LDA', 'NMF']
valid_dataset = ['crisis_12', '20news']

#TODO: add a dir here, maybe using a class is easier
save_dir_crisis_12 = '/content/drive/MyDrive/SS_2022_Praktikum/Crisis Dataset/Dataset_12'

#TODO: solve random state problem (we may need multiple runs to get average)
def runner(model_name, dataset_name, top_n_topics,
           n_words, save_dir, random_state = 98):
    
    # Make some assertions
    if model_name not in valid_models:
        raise ValueError(f'{model_name} not in valid embedding models: {valid_models}.')

    if dataset_name not in valid_dataset:
        raise ValueError(f'{dataset_name} not in valid embedding models: {valid_dataset}.')
    
    
    # Define dataset
    dataset = Dataset()
    if dataset_name == 'crisis_12':
        dataset.load_custom_dataset_from_folder(save_dir)
        num_topics = 4
    elif dataset_name == '20news':
        dataset.fetch_dataset("20NewsGroup")
        num_topics = 20

    # Document start time and end time to get the time the model has used    
    start_time = time.time()

    if model_name == 'LDA':
        OCTIS_model = LDA(num_topics = num_topics, random_state = random_state)        # Create the model
        output = OCTIS_model.train_model(dataset)                                      # Train the model

    elif model_name == 'NMF':
        OCTIS_model = NMF(num_topics = num_topics, random_state = random_state)        # Create the model
        output = OCTIS_model.train_model(dataset)                                      # Train the model
    
    end_time = time.time()
    run_time = round(end_time-start_time, 2)

    # Initialize topic coherence metric
    npmi = Coherence(texts=dataset.get_corpus(), topk=10, measure='c_v')
    # Initialize topic diversity metric
    topic_diversity = TopicDiversity(topk=10)
    
    # Get the TC and TD scores
    npmi_score = npmi.score(output)
    topic_diversity_score = topic_diversity.score(output)
    
    # To match the real label of crisis_12, we need a dict
    #TODO: maybe we can delete it
    in_labels_crisis_12 = {0:'earthquake',
                           1:'floods',
                           2:'forestfires',
                           3:'hurricanes'}
    
    #TODO: fit this to 20news, currently only crisis_12
    # Get Doc-Topic Output
    col_names_DT = ('run_id', 'Document ID', 'Document', 'Real Label', 'Assigned Topic Num', 'Assignment Score')
    result_DT = []
        
    # Dataset 12
    df = pd.read_csv('/content/drive/MyDrive/SS_2022_Praktikum/Crisis Dataset/Dataset_12/corpus_with_header.tsv', delimiter='\t')
        
    # Crisis dataset
    df = df.loc[df['partition'] == 'train']
    output_max = np.max(output['topic-document-matrix'], axis = 0)
    output_index = np.argmax(output['topic-document-matrix'], axis = 0)
    length = len(output_max)
        
    for i in range(length):
        result_DT_line = []
        
        result_DT_line.append(start_time)
        result_DT_line.append(i)
        result_DT_line.append(df[i:i+1].Tweets.values[0])
        result_DT_line.append(in_labels_crisis_12[df[i:i+1].Topics.values[0]])
        result_DT_line.append(output_index[i])
        result_DT_line.append(output_max[i])
        result_DT.append(result_DT_line)
        
    Doc_Topic_df = pd.DataFrame(result_DT, columns = col_names_DT)

    #TODO: same for this, just as Doc-Topic    
    # Get Topic-Word Output
    col_names_TW = ("run_id", "method", "method_specific_params", "dataset", "num_given_topics", "reduced",
                    "topic_num", "topic_size", "topic_words", "word_scores", "num_detected_topics",
                    "num_final_topics", "duration_secs")
    result_TW = []
    size_count = Doc_Topic_df.loc[:, 'Assigned Topic Num'].value_counts()
    topic_words = output['topics']
        
    # Get word scores
    scores = output['topic-word-matrix']
        
    word_scores_all = []
    for i in range(4):
        scores[i].sort()
        word_scores_line = scores[i][-10:][::-1]
        word_scores_all.append(word_scores_line)
        
    for i in range(4):
        result_TW_line = []
        
        result_TW_line.append(start_time)
        result_TW_line.append(model_name)
        result_TW_line.append('None')
        result_TW_line.append(dataset_name)
        result_TW_line.append(4)
        result_TW_line.append('None')
        result_TW_line.append(i)
        result_TW_line.append(size_count[i])
        result_TW_line.append(topic_words[i])
        result_TW_line.append(word_scores_all[i])
        result_TW_line.append('None')
        result_TW_line.append('None')
        result_TW_line.append(run_time)
        result_TW.append(result_TW_line)
        
    Topic_word_df = pd.DataFrame(result_TW, columns=col_names_TW)

    return Doc_Topic_df, Topic_word_df



#TODO: Add TC and TD output
#TODO: Add all_run (4 combinations for now)


if __name__ == '__main__':
    Doc_Topic_df, Topic_word_df = runner(model_name = settings_example['model'], 
                                         dataset_name = settings_example['dataset'], 
                                         top_n_topics = settings_example['top_n_topics'], 
                                         n_words = settings_example['n_words'],
                                         save_dir = save_dir_crisis_12,
                                         random_state = 100)
























































