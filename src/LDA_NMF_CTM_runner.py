import os
import time
from typing import List

import nltk
import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from nltk.tokenize import word_tokenize
from octis.dataset.dataset import Dataset
from octis.models.CTM import CTM
from octis.models.LDA import LDA
from octis.models.NMF import NMF
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

EMBEDDING_DIR_PATH = './pretrained_models'
HUGGING_FACE_EMBEDDING_MODELS = ['bert-base-nli-mean-tokens', "all-mpnet-base-v2", "all-distilroberta-v1",
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


def df_to_vocab_old(df):
    import itertools
    word_list_of_lists = [word_tokenize(df.Text[i]) for i in range(df.shape[0])]
    return list(set(list(itertools.chain.from_iterable(word_list_of_lists))))


def create_vocabulary(documents, save_to: str = None) -> List[str]:
    dataset = [nltk.word_tokenize(text) for text in documents]
    vocab = Dictionary(documents=dataset)
    vocab.filter_extremes(no_below=1, no_above=1.0, keep_n=2000)
    word_list = [key for key in vocab.token2id.keys() if len(key) > 1]
    if save_to:
        with open(f"{save_to}/vocabulary.txt", 'w') as txt:
            for i in range(len(word_list)):
                txt.write(word_list[i] + '\n')
    return word_list


def make_new_documents(documents, new_vocab: List[str]):
    new_documents = []
    for doc in documents:
        new_sen = [word for word in nltk.word_tokenize(doc) if word in set(new_vocab)]
        new_documents.append(' '.join(new_sen).strip())
    return new_documents


def create_dataset(df: pd.DataFrame, vocab: List[str], multilabel: bool = False) -> Dataset:
    df = df.copy()
    dataset_metadata = dict()
    dataset_labels = None

    if len(df.keys()) > 1:
        # just make sure docs are sorted in the right way (train - val - test)
        df.columns = np.arange(len(df.columns))  # Remove column names
        df = pd.concat([df[df[1] == 'train'], df[df[1] == 'val'], df[df[1] == 'test']])
        dataset_metadata['last-training-doc'] = len(df[df[1] == 'train'])
        dataset_metadata['last-validation-doc'] = len(df[df[1] == 'val']) + len(df[df[1] == 'train'])

        if len(df.keys()) > 2:
            dataset_labels = [doc.split() for doc in df[2].tolist()] if multilabel else df[2].tolist()
    else:
        dataset_metadata['last-training-doc'] = len(df[0])

    dataset_corpus = [d.split() for d in df[0].tolist()]
    return Dataset(corpus=dataset_corpus, vocabulary=vocab, labels=dataset_labels, metadata=dataset_metadata)


def runner(args, model_name: str, run_id: int, output_folder: str):
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    rnd_state = args['random_state']

    if model_name == 'ctm':
        assert args["embedding_model"] in HUGGING_FACE_EMBEDDING_MODELS, \
            f'"{args["embedding_model"]}" must be in {HUGGING_FACE_EMBEDDING_MODELS}!'
        download_embedding_models(embedding_folder=EMBEDDING_DIR_PATH)

    start_time = time.time()  # Document start time and end time to get the time the model has used

    # First, make vocabulary (words restricted to 2000) and the short document
    vocabulary = create_vocabulary(args['docs'], save_to=output_folder)
    documents = make_new_documents(args['docs'], vocabulary)

    # Make dataset for LDA/NMF. These two algorithms don't need val and test set.
    if model_name == 'lda':
        method_specific_parameters = {
            'alpha': args['alpha'],
        }
        result_all = [[doc, 'train', label] for doc, label in zip(documents, args['labels'])]
        octis_model = LDA(num_topics=args['num_topics'], random_state=rnd_state,
                          **method_specific_parameters)  # Create the model
    elif model_name == 'nmf':
        method_specific_parameters = {}  # No method specific parameters now
        result_all = [[doc, 'train', label] for doc, label in zip(documents, args['labels'])]
        octis_model = NMF(num_topics=args['num_topics'], random_state=rnd_state,
                          **method_specific_parameters)  # Create the model
    elif model_name == 'ctm':  # TODO: CTM does not guarantee reproducibility, we may need multiple runs to get average
        method_specific_parameters = {
            'num_epochs': args['num_epochs'],
            'lr': args['learning_rate'],
            'batch_size': args['batch_size']
        }
        df_labels = pd.DataFrame(args['labels'])
        pt_test, pt_val = 0.1, 0.1  # TODO: maybe we can play with them
        pt_test_val = pt_test + pt_val
        y_train, y_tst = train_test_split(df_labels.copy(), stratify=df_labels, test_size=pt_test_val, random_state=rnd_state)
        y_tst, y_val = train_test_split(y_tst, stratify=y_tst, test_size=pt_val / pt_test_val, random_state=rnd_state)
        y_train['partition'] = 'train'
        y_tst['partition'] = 'test'
        y_val['partition'] = 'val'
        partitions = pd.concat([y_train, y_tst, y_val]).sort_index()['partition']
        result_all = zip(documents, partitions, args['labels'])
        octis_model = CTM(bert_path=f'{output_folder}/ctm',
                          bert_model=f'{EMBEDDING_DIR_PATH}/sentence-transformers_{args["embedding_model"]}',
                          num_topics=args['num_topics'],
                          **method_specific_parameters
                          )  # todo: add hyperparams

    else:
        raise ValueError(f'Wrong model name, given model_name:"{model_name}", available: {"lda", "nmf", "ctm"}.')

    corpus_df = pd.DataFrame(result_all, columns=('Text', 'Partition', 'Topics'))
    dataset = create_dataset(df=corpus_df, vocab=vocabulary)
    print('[INFO] Model is training..')
    output = octis_model.train_model(dataset)  # Train the model
    print('[INFO] Model trained successfully!')

    end_time = time.time()
    run_time = round(end_time - start_time, 2)

    corpus_df = corpus_df.loc[corpus_df['Partition'] == 'train']
    output_max = np.max(output['topic-document-matrix'], axis=0)
    output_index = np.argmax(output['topic-document-matrix'], axis=0)

    result_DT = []  # Get Doc-Topic Output
    for i in range(len(output_max)):
        result_DT.append([run_id, i, corpus_df[i:i + 1].Text.values[0],
                          corpus_df[i:i + 1].Topics.values[0], output_index[i], output_max[i]])
    df_output_doc_topic = pd.DataFrame(result_DT, columns=('run_id', 'Document ID', 'Document', 'Real Label',
                                                           'Assigned Topic Num', 'Assignment Score'))

    # Get Topic-Word Output
    col_names_TW = ("run_id", "method", "method_specific_params", "dataset", "num_given_topics", "reduced",
                    "topic_num", "topic_size", "topic_words", "word_scores", "num_detected_topics",
                    "num_final_topics", "duration_secs")
    result_TW = []
    size_count = df_output_doc_topic.loc[:, 'Assigned Topic Num'].value_counts()
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
        result_TW_line = [run_id, model_name, method_specific_parameters, args['data_name'], args['num_topics'], False,
                          i,
                          size_count[i], topic_words[i], word_scores_all[i], args['num_topics'], args['num_topics'],
                          run_time]

        result_TW.append(result_TW_line)

    Topic_word_df = pd.DataFrame(result_TW, columns=col_names_TW)

    if os.path.exists('./checkpoint.pt'):  # Delete artifact of pytorch
        os.remove(path='./checkpoint.pt')
    return df_output_doc_topic, Topic_word_df
