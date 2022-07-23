medium_preprocessing_methods = [
    'to_lowercase',
    'standardize_accented_chars',
    'remove_url',
    'expand_contractions',
    'remove_mentions',
    'remove_hashtags',
    # 'remove_new_lines',
    'keep_only_alphabet',
    # 'remove_extra_spaces',
    'remove_english_stop_words',
    'lemmatize_noun'
]
ctm_configs = {
    'dataset': 'crisis_toy',
    'preprocessing_funcs': medium_preprocessing_methods,
    'algorithm_args': {
        'algorithm': 'ctm',
        'num_topics': 4,
        'random_state': 42,
        # 'embedding_model':'bert-base-nli-mean-tokens',
        'embedding_model': 'all-mpnet-base-v2',
        # 'embedding_model': 'doc2vec',
        # 'embedding_model': 'universal-sentence-encoder',
        # 'embedding_model': 'universal-sentence-encoder-large', # WORKS VERY WELL
        # 'embedding_model': 'distiluse-base-multilingual-cased'
        'num_epochs': 100,
        'learning_rate': 2e-3,
        'batch_size': 64,
        'alpha': 'asymmetric',

    }
}

lda_configs = {
    'dataset': 'crisis_toy',
    'preprocessing_funcs': medium_preprocessing_methods,
    'algorithm_args': {
        'algorithm': 'lda',
        'num_topics': 4,
        'random_state': 42,
        'alpha': 'asymmetric'
    }
}

nmf_configs = {
    'dataset': 'crisis_toy',
    'preprocessing_funcs': medium_preprocessing_methods,
    'algorithm_args': {
        'algorithm': 'nmf',
        'num_topics': 4,
        'random_state': 42,
    }
}

bertopic_configs = {
    'dataset': 'crisis_toy',
    'preprocessing_funcs': medium_preprocessing_methods,
    'algorithm_args': {
        'algorithm': 'bertopic',
        # "embedding_model": "all-MiniLM-L6-v2",
        # "embedding_model": "all-distilroberta-v1",
        # "embedding_model": "doc2vec",
        "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
        "top_n_words": 10,
        "n_gram_range_tuple": (1, 1),
        # Both the same as below
        "min_docs_per_topic": 15,
        "num_topics": 4,
        "cluster_model": "hdbscan",  # "hdbscan"or "kmeans" then we do not need hdbscan_args; if used are ignored
        "hdbscan_args": {
            "min_cluster_size": 15,
            "metric": 'euclidean',
            "cluster_selection_method": 'eom',
            "prediction_data": True,
            # "min_samples": 15
        },
        "umap_args": {
            "n_neighbors": 15,
            "n_components": 5,
            "min_dist": 0.0,
            "metric": 'cosine',
            "low_memory": False,
            "random_state": 42
        }
    }
}

lda_bert_configs = {
    'dataset': 'crisis_toy',
    'preprocessing_funcs': medium_preprocessing_methods,
    'algorithm_args': {
        'algorithm': 'lda-bert',
        'embedding_model': "paraphrase-multilingual-MiniLM-L12-v2",
        'num_topics': 4,
        'top_n_words': 10,
        'gamma': 15,
        'random_state': 42
    }
}

top2vec_configs = {
    'dataset': 'crisis_toy',
    'preprocessing_funcs': medium_preprocessing_methods,
    'algorithm_args': {
        'algorithm': 'top2vec',
        'num_topics': 4,
        # 'embedding_model': 'doc2vec',
        'embedding_model': 'universal-sentence-encoder',
        # 'embedding_model': 'universal-sentence-encoder-large', # WORKS VERY WELL
        # 'embedding_model': 'distiluse-base-multilingual-cased',
        'doc2vec_speed': 'learn',
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
        }
    }
}

# configs_list = [lda_bert_configs, nmf_configs, ctm_configs, bertopic_configs, lda_configs, top2vec_configs, ]
configs_list = [lda_bert_configs, nmf_configs]  # , ctm_configs, bertopic_configs, lda_configs,top2vec_configs,  ]
