import os.path
import tarfile
import time
from collections import OrderedDict
from multiprocessing import cpu_count
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from top2vec import Top2Vec

from utils import load_documents, pretty_print_dict

EMBEDDING_DIR_PATH = './pretrained_models'
TF_HUB_EMBEDDING_MODELS_WITH_SOURCES = [
    {'name': 'universal-sentence-encoder',
     'source': 'https://tfhub.dev/google/universal-sentence-encoder/4?tf-hub-format=compressed'},
    {'name': 'universal-sentence-encoder-multilingual',
     'source': 'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3?tf-hub-format=compressed'},
    {'name': 'universal-sentence-encoder-large',
     'source': 'https://tfhub.dev/google/universal-sentence-encoder-large/5?tf-hub-format=compressed'},
    {'name': 'universal-sentence-encoder-multilingual-large',
     'source': 'https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3?tf-hub-format=compressed'},
]
TF_HUB_EMBEDDING_MODELS = [emb_model['name'] for emb_model in TF_HUB_EMBEDDING_MODELS_WITH_SOURCES]
HUGGING_FACE_EMBEDDING_MODELS = ['distiluse-base-multilingual-cased', 'all-MiniLM-L6-v2',
                                 'paraphrase-multilingual-MiniLM-L12-v2']
VALID_EMBEDDING_MODELS = ['doc2vec'] + HUGGING_FACE_EMBEDDING_MODELS + TF_HUB_EMBEDDING_MODELS


def download_embedding_models(embedding_folder: str, remove_tar_gz: bool = True) -> None:
    if not os.path.exists(embedding_folder):
        print(f'[INFO] The embedding folder "{embedding_folder}" download folder was missing, so creating..')
        os.mkdir(embedding_folder)

    for embedding_model in HUGGING_FACE_EMBEDDING_MODELS:
        target_path = f'{embedding_folder}/sentence-transformers_{embedding_model}'
        if not os.path.exists(target_path):
            print(f'[INFO] The embedding model folder:"{target_path}" not found, downloading..')
            SentenceTransformer(model_name_or_path=embedding_model, cache_folder=embedding_folder)
            print(f'[INFO] The embedding model folder:"{target_path}" downloaded.')

    for embedding_model in TF_HUB_EMBEDDING_MODELS_WITH_SOURCES:
        target_path = f'{embedding_folder}/{embedding_model["name"]}'

        if not os.path.exists(target_path):
            print(f'[INFO] The embedding model folder:"{target_path}" not found, downloading..')
            response = requests.get(url=embedding_model["source"], stream=True)
            if response.status_code == 200:
                with open(target_path + '.tar.gz', 'wb') as f:
                    f.write(response.raw.read())
            print(f'[INFO] The embedding model folder:"{target_path}" downloaded.')

            file = tarfile.open(target_path + '.tar.gz')
            print(f'[INFO] Extracting the downloaded embedding model :"{target_path}.tar.gz"..')
            file.extractall(target_path)
            print(f'[INFO] Extracted the downloaded embedding model :"{target_path}.tar.gz".')
            file.close()
            if remove_tar_gz:
                os.remove(path=target_path + '.tar.gz')
                print(f'[INFO] Deleted the downloaded embedding model archive:"{target_path}.tar.gz".')
        else:
            print(f'[INFO] The embedding model folder:"{target_path}" found, so no need to download.')


def print_topic_stats(stats: List[Dict[str, Any]]) -> None:
    for stat in stats:
        print(f'[INFO] Topic #{str(stat["topic_num"]).zfill(2)}:')
        print(f'     > From Reduced Model:{stat["reduced"]}.')
        print(f'     > Topic Size:{stat["topic_size"]}.')
        print(f'     > Topic Words:', str(stat['topic_words']).replace('\n', '\n\t\t'))
        print(f'     > Topic Word Scores:', str(stat['word_scores']).replace('\n', '\n\t\t'))


def get_topic_stats(model_t2v: Top2Vec, is_reduced: bool = False) -> List[Dict[str, Any]]:
    num_topics = model_t2v.get_num_topics(reduced=is_reduced)
    topic_sizes, topic_nums = model_t2v.get_topic_sizes(reduced=is_reduced)
    topic_words, word_scores, _ = model_t2v.get_topics(num_topics=num_topics, reduced=is_reduced)

    stats = []
    for tn, ts, tw, ws in zip(topic_nums, topic_sizes, topic_words, word_scores):
        stats.append(OrderedDict([
            ('reduced', is_reduced),
            ('topic_num', tn),
            ('topic_size', ts),
            ('topic_words', tw),
            ('word_scores', ws)
        ]))
    return stats


def extract_doc_topic_output(run_id: float, topic_stats: List[Dict], model: Top2Vec,
                             labels: List[str], is_reduced: bool) -> pd.DataFrame:
    doc_topic_outputs = []
    for topic_stat in topic_stats:
        docs, doc_scores, document_ids = model.search_documents_by_topic(
            topic_num=topic_stat['topic_num'], num_docs=topic_stat['topic_size'], reduced=is_reduced
        )
        for doc_id, doc, score in zip(document_ids, docs, doc_scores):
            doc_topic_outputs.append({'run_id': run_id,
                                      'Document ID': doc_id, 'Document': doc, 'Real Label': labels[doc_id],
                                      'Assigned Topic Num': topic_stat['topic_num'], 'Assignment Score': score})

    return pd.DataFrame(doc_topic_outputs).sort_values('Document ID')


def extract_topic_word_output(
        run_id: float, topic_stats: List[Dict], method_specific_params: dict, dataset_dir: str, data_col: str,
        num_topics: int, method: str, num_detected_topics: float, num_final_topics: int, duration_secs: float):
    modeling_params_dict = OrderedDict([
        ('run_id', run_id),
        ('method', method),
        ('method_specific_params', method_specific_params),
        ('dataset_name', Path(dataset_dir).name),
        ('data_col', data_col),
        ('num_given_topics', num_topics),
    ])

    modeling_results_dict = OrderedDict([
        ('num_detected_topics', num_detected_topics),
        ('num_final_topics', num_final_topics),
        ('duration_secs', duration_secs),
    ])

    params_df = pd.DataFrame([modeling_params_dict] * len(topic_stats))
    results_df = pd.DataFrame([modeling_results_dict] * len(topic_stats))
    return pd.concat([params_df, pd.DataFrame(topic_stats), results_df], axis=1)


def run(dataset_dir: str, min_count: int, embedding_model: str, umap_args: Dict, hdbscan_args: Dict,
        doc2vec_speed: str = None, num_topics: int = None, data_col: str = None) -> Tuple:
    """
    Runs Top2Vec algorithm with the given parameters.

    Parameters
    ----------
    dataset_dir: Dataset Directory

    min_count:  Set in the Top2Vec paper. Ignores all words with total frequency lower than this. For smaller corpora a
                smaller min_count is necessary. NOTE: This value largely depends on corpus size and its vocabulary.

    embedding_model:    Embedding model for the part where semantic relationships of the data are being learned. Also
                        note that Doc2Vec is not deterministic but other models are deterministic as long as we
                        give random seed to UMAP, source: https://github.com/ddangelov/Top2Vec/issues/86.
                        Options: ['doc2vec', 'universal-sentence-encoder', 'universal-sentence-encoder-large',
                        'universal-sentence-encoder-multilingual', 'universal-sentence-encoder-multilingual-large',
                        'distiluse-base-multilingual-cased', 'all-MiniLM-L6-v2',
                        'paraphrase-multilingual-MiniLM-L12-v2']

    umap_args:
        n_neighbors:    Set in the Top2Vec paper. The size of local neighborhood (in terms of number of neighboring
                        sample points) used for manifold approximation. Larger values result in more global views of the
                        manifold, while smaller values result in more local data being preserved. In general values
                        should be in the range 2 to 100.
        n_components:   Set in the Top2Vec paper. the dimension of the space to embed into. This defaults to 2 to
                        provide easy visualization, but can reasonably be set to any integer value in the range 2 to
                        100.
        metric:     Set in the Top2Vec paper. Options: ['euclidean', 'manhattan', 'chebyshev', 'minkowski',
                    'canberra', 'braycurtis', 'mahalanobis', 'wminkowski', 'seuclidean', 'cosine', 'correlation',
                    'haversine', 'hamming', 'jaccard', 'dice', 'russelrao', 'kulsinski', 'll_dirichlet',
                    'hellinger', 'rogerstanimoto', 'sokalmichener', 'sokalsneath', 'yule'].

    hdbscan_args:
        min_cluster_size:   Set in the Top2Vec paper. The minimum size of clusters; single linkage splits that contain
                            fewer points than this will be considered points "falling out" of a cluster rather than a
                            cluster splitting into two new clusters.
        metric:     Set in the Top2Vec paper. The metric to use when calculating distance between instances in a feature
                    array. If metric is a string or callable, it must be one of the options allowed by
                    metrics.pairwise.pairwise_distances for its metric parameter. If metric is "precomputed", X is
                    assumed to be a distance matrix and must be square. Options: ['cosine', 'euclidean', 'haversine',
                    'l2', 'l1', 'manhattan', 'precomputed', 'nan_euclidean'].
        cluster_selection_method:   Set in the Top2Vec paper. The method used to select clusters from the condensed
                                    tree. The standard approach for HDBSCAN* is to use an Excess of Mass algorithm
                                    to find the most persistent clusters. Alternatively you can instead
                                    select the clusters at the leaves of the tree -- this provides the
                                    most fine-grained and homogeneous clusters. Options: ['eom','leaf'].

    doc2vec_speed:  This parameter is only used when using doc2vec as embedding_model. Options: ['fast-learn',
                    'learn', 'deep-learn']

    num_topics: Given number of topics. If model can reduce the number of topics, it can reduce to num_topics.

    data_col: Data column of the given datasets. For 20newsgroup dataset, it is redundant.
    """
    assert embedding_model in VALID_EMBEDDING_MODELS, f'"{embedding_model}" must be in {VALID_EMBEDDING_MODELS}!'
    download_embedding_models(embedding_folder=EMBEDDING_DIR_PATH)
    time_start = time.time()
    print(f'[INFO] Top2Vec is running for dataset directory:"{dataset_dir}".')
    documents, labels = load_documents(dataset_dir, data_col)

    if embedding_model == 'doc2vec':  # Model is Doc2Vec
        model = Top2Vec(
            documents, speed=doc2vec_speed, workers=cpu_count(), min_count=min_count,
            embedding_model=embedding_model, umap_args=umap_args, hdbscan_args=hdbscan_args
        )
    elif embedding_model in TF_HUB_EMBEDDING_MODELS:  # Model is a TF Hub Model (Universal-Sentence-Encoder)
        model = Top2Vec(
            documents, workers=cpu_count(), min_count=min_count, embedding_model=embedding_model,
            embedding_model_path=f'{EMBEDDING_DIR_PATH}/{embedding_model}', umap_args=umap_args,
            hdbscan_args=hdbscan_args
        )
    elif embedding_model in HUGGING_FACE_EMBEDDING_MODELS:  # Model is a Hugging Face Model (Sentence-Transformer)
        model = Top2Vec(
            documents, workers=cpu_count(), min_count=min_count, embedding_model=embedding_model,
            embedding_model_path=f'{EMBEDDING_DIR_PATH}/sentence-transformers_{embedding_model}', umap_args=umap_args,
            hdbscan_args=hdbscan_args
        )
    else:
        raise ValueError(f'Given "{embedding_model}" not in valid embedding models: {VALID_EMBEDDING_MODELS}.')

    non_reduced_num_topics = model.get_num_topics(reduced=False)
    print(f'[INFO] Original (Non-reduced) Number of Topics: {non_reduced_num_topics}.')
    is_reduced = False
    if num_topics is not None:
        if non_reduced_num_topics > num_topics:
            model.hierarchical_topic_reduction(num_topics=num_topics)
            is_reduced = True
        else:
            print('[WARN] # of topics is pre-specified but non_reduced_num_topics <= num_topics, so not reduced!')
            print(f'   > non_reduced_num_topics:{non_reduced_num_topics}, given num_topics:{num_topics}!')
            time.sleep(3)
    topic_stats = get_topic_stats(model_t2v=model, is_reduced=is_reduced)

    duration_secs = float('%.3f' % (time.time() - time_start))
    print_topic_stats(stats=topic_stats)

    print(f'[INFO] Top2Vec successfully terminated for data:"{dataset_dir}".')

    # Prepare Output
    df_output_doc_topic = extract_doc_topic_output(run_id=int(time_start), topic_stats=topic_stats, model=model,
                                                   labels=labels, is_reduced=is_reduced)
    df_output_topic_word = extract_topic_word_output(
        run_id=int(time_start), topic_stats=topic_stats, dataset_dir=dataset_dir, data_col=data_col,
        num_topics=num_topics, method='top2vec',
        method_specific_params={
            'doc2vec_speed': doc2vec_speed, 'min_count': min_count, 'embedding_model': embedding_model,
            'umap_args': umap_args, 'hdbscan_args': hdbscan_args,
        },
        num_detected_topics=non_reduced_num_topics, num_final_topics=len(topic_stats), duration_secs=duration_secs
    )
    return model, df_output_doc_topic, df_output_topic_word


def parametric_run(args):
    pretty_print_dict(args, info_log='Top2Vec Parameters:')
    return run(**args)


def default_test():
    return parametric_run(args={
        'dataset_dir': './data/crisis_resource_toy',
        'data_col': 'text',
        # 'dataset_dir': './data/crisis_resource_12_labeled_by_paid_workers',
        # 'data_col': 'text',
        # 'dataset_dir': './data/20news_bydate',
        'num_topics': 4,

        # ####### Top2Vec Specific Arguments #########
        # 'embedding_model': 'doc2vec',
        # 'embedding_model': 'universal-sentence-encoder-large',
        'embedding_model': 'all-MiniLM-L6-v2',
        # 'embedding_model': 'distiluse-base-multilingual-cased',
        # 'doc2vec_speed': 'fast-learn',
        'min_count': 50,
        'umap_args': {
            'n_neighbors': 15,
            'n_components': 5,
            'metric': 'cosine',
            'random_state': 42  # Try to always include this for reproducibility, github.com/ddangelov/Top2Vec/issues/86
        },
        'hdbscan_args': {
            'min_cluster_size': 15,
            'metric': 'euclidean',
            'cluster_selection_method': 'eom'
        },
    }
    )


if __name__ == '__main__':
    default_test()
