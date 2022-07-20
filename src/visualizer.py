import itertools
import math
import os.path
from collections import OrderedDict
from math import ceil
from string import Template
from typing import List, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import umap.plot
from bertopic import BERTopic
from matplotlib import pyplot
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics.pairwise import cosine_similarity
from top2vec import Top2Vec

from src.bertopic_runner import LDABERT

target_template = Template('${run_id}_${algorithm}_${visualization_method}.${extension}')

AVAILABLE_TOPIC_MODELING_ALGORITHMS = ['top2vec', 'bertopic', 'lda', 'nmf', 'lda-bert']

ALGORITHM_TO_WORD_SCORE_METRIC = {
    'top2vec': 'CosineSimilarity(TopicVec,WordVec)',
    'bertopic': 'c-TF-IDF Score',
    'lda': 'Probability Score',
    'nmf': 'Probability Score',
    'lda-bert': 'Word Frequency'
}
assert set(AVAILABLE_TOPIC_MODELING_ALGORITHMS) == set(ALGORITHM_TO_WORD_SCORE_METRIC)


def check_algorithm(al: str):
    assert al in AVAILABLE_TOPIC_MODELING_ALGORITHMS, f'{al} is not available in {AVAILABLE_TOPIC_MODELING_ALGORITHMS}!'


def draw_umap2d_scatter_plot(
        model: Union[Top2Vec, BERTopic, LDABERT],
        df_output_topic_word: pd.DataFrame,
        df_output_doc_topic: pd.DataFrame = None,
        target_dir: str = './output/visualization'
) -> pyplot.Figure:
    run_id = df_output_topic_word['run_id'][0]
    algorithm_name = df_output_topic_word['method'][0]
    check_algorithm(al=algorithm_name)

    if algorithm_name == 'top2vec':
        doc_topics = model.doc_top_reduced if df_output_topic_word['reduced'][0] else model.doc_top
        doc_vectors = model.document_vectors
    elif algorithm_name == 'bertopic':
        doc_topics = df_output_doc_topic['Assigned Topic Num']
        doc_vectors = model.umap_model.embedding_
    elif algorithm_name == 'lda-bert':
        doc_topics = df_output_doc_topic['Assigned Topic Num']
        doc_vectors = model.vec['lda-bert']
    elif algorithm_name in ('lda', 'nmf'):
        raise ValueError(f'LDA and NMF cannot support draw_umap_2d_scatter_plot() because they do not use UMAP phase.')
    else:
        raise ValueError(f'Algorithm is not supported:{algorithm_name} for draw_umap_2d_scatter_plot().')

    if not os.path.exists(target_dir):
        print(f'[INFO] The target dir:"{target_dir}" not exists, so creating..')
        os.makedirs(target_dir)

    # LDA-Bert does not use UMAP for dimension reduction (as of now), so just use fixed args
    if algorithm_name == 'lda-bert':
        visualization_umap_args = {
            "n_neighbors": 15,
            "n_components": 5,
            "min_dist": 0.0,
            "metric": 'cosine',
            "low_memory": False
        }
    else:
        visualization_umap_args = df_output_topic_word['method_specific_params'][0]['umap_args']

    visualization_umap_args.update({'n_components': 2})

    print(f'[INFO] UMAP Arguments for Visualization:{visualization_umap_args}')

    print('[INFO] UMAP Model is being fitted..')
    umap_model = umap.UMAP(**visualization_umap_args).fit(doc_vectors)
    print('[INFO] UMAP Model successfully fitted.')

    target_figure = umap.plot.points(
        umap_model, labels=doc_topics, width=2 * 10 ** 3, height=2 * 10 ** 3,
        theme="viridis").get_figure()  # type: pyplot.Figure
    target_image_filename = target_template.substitute(
        run_id=run_id, algorithm=algorithm_name, visualization_method='umap2d', extension='png')

    target_figure.savefig(fname=f'{target_dir}/{target_image_filename}')
    return target_figure


def visualize_barchart(df_output_topic_word: pd.DataFrame,  # todo: rename the function
                       topics: List[int] = None,
                       top_n_topics: int = None,
                       n_words: int = 5,
                       width: int = 250,
                       height: int = 250) -> go.Figure:
    """ Visualize a barchart of selected topics

    Arguments:
        df_output_topic_word: Output from Algorithm Part
        topics: A selection of topics to visualize.
        top_n_topics: Only select the top n most frequent topics.
        n_words: Number of words to show in a topic
        width: The width of each figure.
        height: The height of each figure.

    Returns:
        fig: A plotly figure
    """
    run_id = df_output_topic_word['run_id'][0]
    algorithm_name = df_output_topic_word['method'][0]
    check_algorithm(al=algorithm_name)
    colors = itertools.cycle(["#D55E00", "#0072B2", "#CC79A7", "#E69F00", "#56B4E9", "#009E73", "#F0E442"])

    if topics is not None:  # Get selected topics
        topics = sorted(topics)
    elif top_n_topics is not None and 0 < top_n_topics < len(df_output_topic_word):  # Get top n topics
        topics = sorted(df_output_topic_word['topic_num'].to_list())[:top_n_topics]
    else:  # Get all topics
        topics = sorted(df_output_topic_word['topic_num'].to_list())

    if algorithm_name == 'bertopic' and -1 in topics:
        topics.remove(-1)

    topic_num_to_df = df_output_topic_word.set_index("topic_num").to_dict("index")

    # Initialize figure
    subplot_titles = [f"Topic {topic}" for topic in topics]
    columns = 4
    rows = int(np.ceil(len(topics) / columns))
    fig = make_subplots(rows=rows,
                        cols=columns,
                        shared_xaxes=False,
                        horizontal_spacing=.1,
                        vertical_spacing=.4 / rows if rows > 1 else 0,
                        subplot_titles=subplot_titles)

    # Add barchart for each topic
    cur_row = 1
    cur_col = 1
    for topic in topics:
        words = [word + "  " for word in topic_num_to_df[topic]['topic_words']][:n_words][::-1]
        scores = [score for score in topic_num_to_df[topic]['word_scores']][:n_words][::-1]

        # noinspection PyTypeChecker
        fig.add_trace(
            go.Bar(x=scores,
                   y=words,
                   orientation='h',
                   marker_color=next(colors)),
            row=cur_row, col=cur_col)

        fig.update_xaxes(title_text=ALGORITHM_TO_WORD_SCORE_METRIC[algorithm_name],
                         title_font=dict(size=13, color="Black"), title_standoff=0, row=cur_row, col=cur_col)

        if cur_col == columns:
            cur_col = 1
            cur_row += 1
        else:
            cur_col += 1

    # Stylize graph
    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        title={
            'text': f'<b>Topic Word Scores for algorithm="{algorithm_name}" and run_id="{run_id}"',
            'x': .5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        },
        width=width * 4,
        height=height * rows if rows > 1 else height * 1.3,
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
    )

    fig.update_xaxes(showgrid=True, range=[0, 1], dtick=0.2)
    fig.update_yaxes(showgrid=True)

    # fig = fig # type: plotly.graph_objs._figure.Figure
    # fig.write_image('asd.png') # todo: write as image
    return fig


def visualize_labels_per_topic(df_output_doc_topic: pd.DataFrame,
                               df_output_topic_word: pd.DataFrame,
                               top_n_topics: int = 10,
                               top_n_labels: int = None,
                               topics: List[int] = None,
                               use_normalized_frequency: bool = True,
                               width: int = 1000,
                               height: Union[int, str] = 'adjustable') -> go.Figure:
    """ Visualize topics per class

    Arguments:
        df_output_doc_topic: Doc<->Topic Output Dataframe
        df_output_topic_word: Topic<->Word Output Dataframe
        top_n_topics: To visualize the most frequent topics instead of all
        top_n_labels: To visualize the most frequent labels (per topic) instead of all
        topics: Select which topics you would like to be visualized
        use_normalized_frequency: Whether to per cent normalize each topic's frequency individually
        width: The width of the figure.
        height: The height of the figure.

    Returns:
        A plotly.graph_objects.Figure including all traces

    """

    num_real_labels = len(df_output_doc_topic['Real Label'].unique())
    if top_n_labels is None or top_n_labels > num_real_labels:
        top_n_labels = num_real_labels
    colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#D55E00", "#0072B2", "#CC79A7"]
    run_id = df_output_topic_word['run_id'][0]
    algorithm_name = df_output_topic_word['method'][0]
    check_algorithm(al=algorithm_name)

    if topics is not None:  # Get selected topics
        topics = sorted(topics)
    elif top_n_topics is not None and 0 < top_n_topics < len(df_output_topic_word):  # Get top n topics
        topics = sorted(df_output_topic_word['topic_num'].to_list())[:top_n_topics]
    else:  # Get all topics
        topics = sorted(df_output_topic_word['topic_num'].to_list())

    # Prepare data
    freq_df = df_output_doc_topic[['Real Label', 'Assigned Topic Num']].value_counts()
    freq_df = freq_df.rename('Frequency').reset_index().sort_values('Real Label')
    freq_df.rename(columns={'Real Label': 'Class', 'Assigned Topic Num': 'Topic'}, inplace=True)
    df_output_topic_word['5_words'] = df_output_topic_word['topic_words'].apply(lambda ws: '_'.join(ws[:5]))
    df_output_topic_word['Name'] = df_output_topic_word['topic_num'].astype(str) + '_' + df_output_topic_word['5_words']
    topic_num_to_names = df_output_topic_word.set_index('topic_num').to_dict()['Name']
    freq_df['Name'] = freq_df['Topic'].map(topic_num_to_names)
    data = freq_df.loc[freq_df['Topic'].isin(topics), :]

    # Initialize figure
    title_x_axis = "Normalized Frequency (%)" if use_normalized_frequency else "Frequency (Count)"

    subplot_titles = [f'Topic: "{topic_num_to_names[topic]}"' for topic in list(topics)]
    subplot_titles = [title[:45] + '...' if len(title) > 48 else title for title in subplot_titles]  # Trim Names
    num_columns = 2
    num_rows = ceil((len(topics)) / num_columns)
    rows = int(np.ceil(len(topics) / num_columns))
    fig = make_subplots(rows=rows,
                        cols=num_columns,
                        shared_xaxes=False,
                        horizontal_spacing=.2,
                        vertical_spacing=.6 / rows if rows > 1 else 0,
                        subplot_titles=subplot_titles,
                        )

    cur_row = 1
    cur_col = 1
    for index, topic in enumerate(topics):
        cur_data = data.loc[data.Topic == topic, :]  # type: pd.DataFrame
        cur_data['NormalizedFrequency'] = 100 * cur_data.Frequency / sum(cur_data.Frequency)
        cur_data = cur_data.sort_values(by='Frequency', ascending=False).head(top_n_labels).iloc[::-1]
        cur_data = cur_data.sort_values(by='Class', ascending=False)  # Sort again by labels alphabetically

        if use_normalized_frequency:
            fig.add_trace(
                go.Bar(y=cur_data.Class, x=cur_data['NormalizedFrequency'], marker_color=colors[index % 7],
                       orientation="h"), row=cur_row, col=cur_col)
            fig.update_xaxes(range=[1, 100], dtick=10)
        else:
            fig.add_trace(
                go.Bar(y=cur_data.Class, x=cur_data['Frequency'], marker_color=colors[index % 7], orientation="h"),
                row=cur_row, col=cur_col)

        fig.update_xaxes(title_text=title_x_axis, title_standoff=0, row=cur_row, col=cur_col)

        if use_normalized_frequency:
            for label_class in cur_data.Class.unique():
                fig.add_annotation(
                    y=label_class,
                    text=str(cur_data[cur_data['Class'] == label_class]['Frequency'].values[0]),
                    bgcolor='black', font={'color': 'white'},
                    opacity=0.75,
                    showarrow=False, row=cur_row, col=cur_col
                )

        if cur_col == 1:
            fig.update_yaxes(title_text="Real Label", row=cur_row, col=cur_col)

        if cur_col == num_columns:
            cur_col = 1
            cur_row += 1
        else:
            cur_col += 1

    # Styling of the visualization
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    fig.update_layout(
        showlegend=False,
        title={
            'text': f'<b>Labels per Topic for algorithm="{algorithm_name}", run_id="{run_id}"<br></b>',
            'x': 0.40,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        },
        template="simple_white",
        width=width,
        height=480 + 25 * num_rows * top_n_labels if height == 'adjustable' else height,
    )
    return fig


def visualize_heatmap(
        model: Union[Top2Vec, BERTopic, LDABERT],
        df_output_doc_topic: pd.DataFrame,
        df_output_topic_word: pd.DataFrame,
        topics: List[int] = None,
        top_n_topics: int = None,
        n_clusters: int = None,
        width: int = 1000,
        height: int = 1000) -> go.Figure:
    """ Visualize a heatmap of the topic's similarity matrix

    Based on the cosine similarity matrix between topic embeddings,
    a heatmap is created showing the similarity between topics.

    Arguments:
        model: A fitted BERTopic instance.
        df_output_doc_topic: Doc<->Topic Output Dataframe
        df_output_topic_word: Topic<->Word Output Dataframe
        topics: A selection of topics to visualize.
        top_n_topics: Only select the top n most frequent topics.
        n_clusters: Create n clusters and order the similarity matrix by those clusters.
        width: The width of the figure.
        height: The height of the figure.

    Returns:
        fig: A plotly figure

    Usage:

    If you want to save the resulting figure:

    ```python
    fig = topic_model.visualize_heatmap()
    fig.write_html("path/to/file.html")
    ```
    <iframe src="../../getting_started/visualization/heatmap.html"
    style="width:1000px; height: 720px; border: 0px;""></iframe>
    """
    run_id = df_output_topic_word['run_id'][0]
    algorithm_name = df_output_topic_word['method'][0]

    # Select topic embeddings
    if algorithm_name == 'top2vec':
        tpc_embeddings = model.topic_vectors_reduced if df_output_topic_word['reduced'][0] else model.topic_vectors
    elif algorithm_name == 'bertopic':
        if model.topic_embeddings is not None:
            tpc_embeddings = np.array(model.topic_embeddings)
        else:
            tpc_embeddings = model.c_tf_idf
    elif algorithm_name == 'lda-bert':
        tpc_embeddings = np.array(model.cluster_model.cluster_centers_)
    elif algorithm_name in ('lda', 'nmf'):
        raise ValueError(f'LDA and NMF cannot support visualize_heatmap() because they have no topic embeddings.')
    else:
        raise ValueError(f'Algorithm is not supported:{algorithm_name} for draw_umap_2d_scatter_plot().')

    # Select topics based on top_n and topics args
    freq_df = df_output_doc_topic[['Assigned Topic Num', 'Real Label']].rename(columns={'Assigned Topic Num': 'Topic'})
    freq_df = freq_df.groupby('Topic')['Real Label'].count().reset_index(name='Count')
    freq_df = freq_df.loc[freq_df.Topic != -1, :]  # Necessary step for BERTopic
    if topics is not None:
        topics = list(topics)
    elif top_n_topics is not None:
        topics = sorted(freq_df.Topic.to_list()[:top_n_topics])
    else:
        topics = sorted(freq_df.Topic.to_list())

    # Order heatmap by similar clusters of topics
    if n_clusters:
        if n_clusters >= len(set(topics)):
            raise ValueError("Make sure to set `n_clusters` lower than "
                             "the total number of unique topics.")

        tpc_embeddings = tpc_embeddings[[topic + 1 for topic in topics]]
        distance_matrix = cosine_similarity(tpc_embeddings)
        clusters = fcluster(Z=linkage(distance_matrix, 'ward'), t=n_clusters, criterion='maxclust')

        # Extract new order of topics
        mapping = {cluster: [] for cluster in clusters}
        for topic, cluster in zip(topics, clusters):
            mapping[cluster].append(topic)
        mapping = [cluster for cluster in mapping.values()]
        sorted_topics = [topic for cluster in mapping for topic in cluster]
    else:
        sorted_topics = topics

    # Select embeddings
    indices = np.array([topics.index(topic) for topic in sorted_topics])
    tpc_embeddings = tpc_embeddings[indices]
    distance_matrix = cosine_similarity(tpc_embeddings)

    # Create nicer labels
    df_output_topic_word['5_words'] = df_output_topic_word['topic_words'].apply(lambda ws: '_'.join(ws[:5]))
    df_output_topic_word['Name'] = df_output_topic_word['topic_num'].astype(str) + '_' + df_output_topic_word['5_words']
    topic_num_to_names = df_output_topic_word.set_index('topic_num').to_dict(into=OrderedDict)['Name']
    topic_names = [topic_num_to_names[topic] for topic in sorted_topics]
    max_characters = 35
    new_labels = [label if len(label) < max_characters else label[:max_characters - 3] + "..." for label in topic_names]

    fig = px.imshow(distance_matrix, text_auto=".2f",
                    labels=dict(color="Cosine Similarity Score"),
                    x=new_labels,
                    y=new_labels,
                    color_continuous_scale='GnBu',
                    )

    fig.update_layout(
        title={
            'text': f'<b>Topic Similarity Matrix for algorithm="{algorithm_name}", run_id="{run_id}"<br></b>',
            'y': .95,
            'x': 0.55,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        },
        width=width,
        height=height,

        xaxis=go.layout.XAxis(
            tickangle=-45)
    )
    fig.update_layout(showlegend=True)
    fig.update_layout(legend_title_text='Trend')
    # todo: write as image
    return fig


def draw_representative_docs(df_output_doc_topic: pd.DataFrame, top_n_docs: int = 3):
    def get_partial_dfs(a_df: pd.DataFrame, partition_col: str):
        df_topics = []
        for topic in sorted(a_df[partition_col].unique()):
            df_topic = a_df.query(f'`{partition_col}` == {topic}').sort_values(
                by=partition_col, ascending=False)
            if len(df_topic) > top_n_docs:
                df_topic = df_topic.head(n=top_n_docs)
            df_topics.append(df_topic)

        return pd.concat(df_topics)

    def format_color_groups(a_df, colors=('lightblue', 'white')):
        x = a_df.copy()
        for i, factor in enumerate(x['Assigned Topic Num'].unique()):
            x.loc[x['Assigned Topic Num'] == factor, :] = f'background-color: {colors[i % len(colors)]}'

        return x

    df = get_partial_dfs(a_df=df_output_doc_topic, partition_col='Assigned Topic Num')

    df_style = df.style.apply(format_color_groups, axis=None) \
        .set_table_attributes('style="font-size: 17px"') \
        .set_properties(color='black !important', border='1px black solid !important') \
        .set_table_styles([{'selector': 'th', 'props': [('border', '1px black solid !important')]}]) \
        .set_properties(**{'text-align': 'left'}) \
        .hide_index()

    for col in df.columns:
        max_col_size = max(map(lambda x: len(str(x)), list(df[col]) + [str(col)]))
        max_col_size = max_col_size if max_col_size < 110 else 110
        df_style = df_style.set_properties(subset=[col], **{'width': max_col_size * 8})

    # dfi.export(df_style, 'successful_test.png') # todo: export to file
    return df_style
