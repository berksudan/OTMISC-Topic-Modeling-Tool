no_prep_methods = []
medium_prep_methods = [
    'to_lowercase',
    'standardize_accented_chars',
    'remove_url',
    'expand_contractions',
    'remove_mentions',
    'remove_hashtags',
    'keep_only_alphabet',
    'remove_english_stop_words',
    'lemmatize_noun'
]
full_prep_methods = [
    'to_lowercase',
    'standardize_accented_chars',
    'remove_url',
    'expand_contractions',
    'expand_missing_delimiter',
    'remove_mentions',
    'remove_hashtags',
    'remove_new_lines',
    'remove_html_tags',
    'keep_only_alphabet',
    'remove_extra_spaces',
    'remove_english_stop_words',
    'lemmatize_noun',
    'lemmatize_verb',
    'lemmatize_adjective',
]

dataset = 'crisis_17'
num_topics = 10
#num_topics = 10
base_range = {
    'dataset': dataset,
    'preprocessing_funcs': [no_prep_methods, medium_prep_methods, full_prep_methods],

}
ctm_range = {
    'algorithm': 'ctm',
    'random_state': 42,
    'embedding_model': [
        'bert-base-nli-mean-tokens', "all-mpnet-base-v2",
        "all-distilroberta-v1", "all-MiniLM-L12-v2",
        "all-MiniLM-L6-v2", 'paraphrase-multilingual-MiniLM-L12-v2'],
    'num_epochs': list(range(10, 201, 10)),
    'learning_rate': 2e-3,
    'batch_size': [16, 32, 64, 128],
}
lda_range = {
    'algorithm': 'lda',
    'num_topics': num_topics,
    'random_state': 42,
    'alpha': ['asymmetric', 'auto']
}

nmf_range = {
    'algorithm': 'nmf',
    'num_topics': num_topics,
    'random_state': 42,
}

top2vec_range = {
    'algorithm': 'top2vec',
    'num_topics': num_topics,
    'embedding_model': [
        'universal-sentence-encoder',
        'universal-sentence-encoder-multilingual',
        'universal-sentence-encoder-large',
        'universal-sentence-encoder-multilingual-large',
        'all-MiniLM-L6-v2',
        'paraphrase-multilingual-MiniLM-L12-v2'
    ],
    'doc2vec_speed': 'learn',
    'min_count': [10, 20, 30, 40, 50],
    'umap_args': {
        'n_neighbors': 15,
        'n_components': [3, 5, 7],
        'metric': 'cosine',
        'random_state': 42  # Try to always include this for reproducibility, github.com/ddangelov/Top2Vec/issues/86
    },
    'hdbscan_args': {
        'min_cluster_size': [5, 10, 15],
        'metric': 'euclidean',
        'cluster_selection_method': 'eom'
    }
}
bertopic_configs = {
    'algorithm': 'bertopic',
    "embedding_model": ["all-mpnet-base-v2", "all-distilroberta-v1",
                        "all-MiniLM-L12-v2", "all-MiniLM-L6-v2", 'paraphrase-multilingual-MiniLM-L12-v2']

    ,
    "top_n_words": [5, 10, 15],
    "n_gram_range_tuple": (1, 1),
    # Both the same as below
    "min_docs_per_topic": 15,
    "num_topics": num_topics,
    "cluster_model": "hdbscan",  # "hdbscan"or "kmeans" then we do not need hdbscan_args; if used are ignored
    "hdbscan_args": {
        "min_cluster_size": [5, 10, 15],
        "metric": 'euclidean',
        "cluster_selection_method": 'eom',
        "prediction_data": True,
        # "min_samples": 15
    },
    "umap_args": {
        "n_neighbors": 15,
        "n_components": [3, 5, 7],
        "min_dist": 0.0,
        "metric": 'cosine',
        "low_memory": False,
        "random_state": 42
    }
}

lda_bert_configs = {
    'embedding_model': ["all-mpnet-base-v2", "all-distilroberta-v1",
                        "all-MiniLM-L12-v2", "all-MiniLM-L6-v2", 'paraphrase-multilingual-MiniLM-L12-v2'],
    'num_topics': num_topics,
    'top_n_words': [5, 10, 15],
    'gamma': [5, 10, 15],
    'random_state': 42
}
# alg_ranges = [nmf_range,ctm_range]
alg_ranges = [top2vec_range,
              bertopic_configs,
              lda_bert_configs,
              lda_range,
              nmf_range,
              ctm_range ]
# alg_ranges = [top2vec_range]
