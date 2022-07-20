from octis.evaluation_metrics.diversity_metrics import TopicDiversity, InvertedRBO
from octis.evaluation_metrics.coherence_metrics import Coherence
from sklearn.metrics.cluster import rand_score

import pandas as pd


def compute_topic_scores(df_output_doc_topic: pd.DataFrame,
                         df_output_topic_word: pd.DataFrame) -> pd.DataFrame:
    """ Compute all scores for a topic model

    Arguments:
        df_output_doc_topic: Doc<->Topic Output Dataframe
        df_output_topic_word: Topic<->Word Output Dataframe

    Returns:
        A pd.DataFrame: df_output_topic_word with additional metric columns 

    """

    scores = {}
    model_output = {}

    if df_output_topic_word['method'].unique() == 'bertopic':
        ## Exclude noise cluster for bertopic
        model_output['topics'] = df_output_topic_word[~df_output_topic_word["topic_num"].isin([-1])]["topic_words"].tolist()
    else:
        model_output['topics'] = df_output_topic_word["topic_words"].tolist()


    scorer_diversity_unique = TopicDiversity(topk=10)
    scores['diversity_unique'] = scorer_diversity_unique.score(model_output)

    scorer_diversity_inverted_rbo = InvertedRBO(topk=10)
    scores['diversity_inv_rbo'] = scorer_diversity_inverted_rbo.score(model_output)

    docs = df_output_doc_topic["Document"].tolist()

    if not all(isinstance(d, list) for d in docs):
        docs = [d.split(' ') for d in docs]

    scorer_coherence_npmi = Coherence(texts=docs, topk=10, measure='c_npmi')
    scores['coherence_npmi'] = scorer_coherence_npmi.score(model_output)

    scorer_coherence_v = Coherence(texts=docs, topk=10, measure='c_v')
    scores['coherence_v'] = scorer_coherence_v.score(model_output)

    scores['rand_index'] = rand_score(df_output_doc_topic["Real Label"].tolist(),
                                      df_output_doc_topic["Assigned Topic Num"].tolist())

    df_metrics = pd.DataFrame([scores] * df_output_topic_word.shape[0])

    df_final = pd.concat([df_output_topic_word, df_metrics], axis=1)

    return df_final
