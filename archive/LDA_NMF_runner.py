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

from gensim.corpora import Dictionary

from src import visualizer

def df_to_vocab_old(df):
    import itertools
    word_list_of_lists = [word_tokenize(df.Text[i]) for i in range(df.shape[0]) ]
    return list(set(list(itertools.chain.from_iterable(word_list_of_lists))))

# The following three functions are used to make vocabulary.txt

def make_word_list(documents):
    dataset = []
    for text in documents:
        tokens = nltk.word_tokenize(text)
        dataset.append(tokens)
    vocab = Dictionary(documents=dataset, prune_at=20000)
    vocab.filter_extremes(no_below=1, no_above=1, keep_n=2000)
    word_list = []
    for key in vocab.token2id.keys():
        if len(key) > 1 or key == 'a':
            word_list.append(key)
    return word_list 

def make_new_documents(documents, word_list):
    for i in range(len(documents)):
        new_sen = []
        for word in nltk.word_tokenize(documents[i]):
            if word in word_list:
                new_sen.append(word)
        documents[i] = ' '.join(new_sen).strip()
    return documents

def list_to_vocab(word_list, save_dir):
    txt = open(f"{save_dir}/vocabulary.txt", 'w')
    for i in range(len(word_list)):
        txt.write(word_list[i])
        txt.write('\r\n')
    txt.close()

#TODO: solve random state problem (we may need multiple runs to get average)
def runner(args):

    word_list = make_word_list(args['docs'])
    documents = make_new_documents(args['docs'], word_list)
    list_to_vocab(word_list, args['data_save_dir'])
    
    # Fit the data into OCTIS format, so that OCTIS can read it
    col_names = ('Text', 'Partition', 'Topics')
    result_all = []

    for i in range(len(args['docs'])):
        if len(args['docs'][i].split()) >= 3:     # Delete data that cannot be recognized by OCTIS
            result = []
            result.append(documents[i])
            result.append('train')
            result.append(args['labels'][i])
            result_all.append(result)

    df_dataset = pd.DataFrame(result_all, columns = col_names)
    df_dataset = df_dataset.sample(frac=1.0, random_state = 100).reset_index(drop=True)
    df_dataset.to_csv(f"{args['data_save_dir']}/corpus.tsv",\
                      sep = '\t', index=False, header = None)
    df_dataset.to_csv(f"{args['data_save_dir']}/corpus_with_header.tsv",\
                      sep = '\t', index=False)

    # The data is now ready, let's start training!
    dataset = Dataset()
    dataset.load_custom_dataset_from_folder(args['data_save_dir'])

    # Document start time and end time to get the time the model has used    
    start_time = time.time()

    if args['model_name'] == 'lda':
        OCTIS_model = LDA(num_topics = args['num_topics'], random_state = args['random_state'])        # Create the model
        output = OCTIS_model.train_model(dataset)                                                      # Train the model

    elif args['model_name'] == 'nmf':
        OCTIS_model = NMF(num_topics = args['num_topics'], random_state = args['random_state'])        # Create the model
        output = OCTIS_model.train_model(dataset)                                                      # Train the model
    
    end_time = time.time()
    run_time = round(end_time-start_time, 2)
    
    #TODO: fit this to 20news, currently only crisis_12
    # Get Doc-Topic Output
    col_names_DT = ('run_id', 'Document ID', 'Document', 'Real Label', 'Assigned Topic Num', 'Assignment Score')
    result_DT = []
        
    df = pd.read_csv(args['data_save_dir'] + '/corpus_with_header.tsv', delimiter='\t')
        
    # Crisis dataset
    df = df.loc[df['Partition'] == 'train']
    output_max = np.max(output['topic-document-matrix'], axis = 0)
    output_index = np.argmax(output['topic-document-matrix'], axis = 0)
    #print(output_index)
    length = len(output_max)
        
    for i in range(length):
        result_DT_line = []
        
        result_DT_line.append(int(start_time))
        result_DT_line.append(i)
        result_DT_line.append(df[i:i+1].Text.values[0])
        result_DT_line.append(df[i:i+1].Topics.values[0])
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
    for i in range(args['num_topics']):
        scores[i].sort()
        word_scores_line = scores[i][-10:][::-1]
        word_scores_line /= np.sum(word_scores_line)
        word_scores_all.append(word_scores_line)
        
    for i in range(args['num_topics']):
        result_TW_line = []
        
        result_TW_line.append(int(start_time))
        result_TW_line.append(args['model_name'])
        result_TW_line.append('None')
        result_TW_line.append(args['data_name'])
        result_TW_line.append(args['num_topics'])
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

# TODO: Do we need that?
#if __name__ == '__main__':
#    Doc_Topic_df, Topic_word_df = runner(model_name = settings_example['model'], 
#                                         dataset_name = settings_example['dataset'], 
#                                         top_n_topics = settings_example['top_n_topics'], 
#                                         n_words = settings_example['n_words'],
#                                         save_dir = save_dir_crisis_12,
 #                                        random_state = 100)
























































