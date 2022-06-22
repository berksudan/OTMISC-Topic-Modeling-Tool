import itertools
import os.path
from string import Template
from typing import List
from typing import Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import umap.plot
from bertopic import BERTopic
from matplotlib import pyplot
from plotly.subplots import make_subplots
from top2vec import Top2Vec

target_template = Template('${run_id}_${algorithm}_${visualization_method}.${extension}')

AVAILABLE_TOPIC_MODELING_ALGORITHMS = ['top2vec', 'bertopic', 'lda', 'nmf']


def draw_umap2d_scatter_plot(
        algorithm: str, model: Union[Top2Vec, BERTopic], df_output_topic_word: pd.DataFrame,
        target_dir: str = './output/visualization') -> pyplot.Figure:
    if algorithm == 'top2vec':
        doc_topics = model.doc_top_reduced if df_output_topic_word['reduced'][0] else model.doc_top
        doc_vectors = model.document_vectors
    elif algorithm == 'bertopic':
        raise NotImplementedError(f'draw_umap_2d_scatter_plot() not implemented for the algorithm:{algorithm}.')
    elif algorithm == 'lda':
        raise NotImplementedError(f'draw_umap_2d_scatter_plot() not implemented for the algorithm:{algorithm}.')
    elif algorithm == 'nmf':
        raise NotImplementedError(f'draw_umap_2d_scatter_plot() not implemented for the algorithm:{algorithm}.')
    else:
        raise ValueError(f'Algorithm is not supported:{algorithm} for draw_umap_2d_scatter_plot().')

    assert algorithm in AVAILABLE_TOPIC_MODELING_ALGORITHMS, \
        f'{algorithm} is not available in {AVAILABLE_TOPIC_MODELING_ALGORITHMS}!'
    if not os.path.exists(target_dir):
        print(f'[INFO] The target dir:"{target_dir}" not exists, so creating..')
        os.makedirs(target_dir)

    run_id = df_output_topic_word['run_id'][0]

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
        run_id=run_id, algorithm=algorithm, visualization_method='umap2d', extension='png')

    target_figure.savefig(fname=f'{target_dir}/{target_image_filename}')
    return target_figure


def visualize_barchart(algorithm: str,
                       df_output_topic_word: pd.DataFrame,
                       topics: List[int] = None,
                       top_n_topics: int = None,
                       n_words: int = 5,
                       width: int = 250,
                       height: int = 250) -> go.Figure:
    """ Visualize a barchart of selected topics

    Arguments:
        df_output_topic_word: Output from Algorithm Part
        algorithm: Name of the Algorithm
        topics: A selection of topics to visualize.
        top_n_topics: Only select the top n most frequent topics.
        n_words: Number of words to show in a topic
        width: The width of each figure.
        height: The height of each figure.

    Returns:
        fig: A plotly figure

    Usage:

    if you want to save the resulting figure:

    ```python
    fig = topic_model.visualize_barchart()
    fig.write_html("path/to/file.html")
    ```
    <iframe src="../../getting_started/visualization/bar_chart.html"
    style="width:1100px; height: 660px; border: 0px;""></iframe>

    Parameters
    ----------

    """
    assert algorithm in AVAILABLE_TOPIC_MODELING_ALGORITHMS, \
        f'{algorithm} is not available in {AVAILABLE_TOPIC_MODELING_ALGORITHMS}!'
    colors = itertools.cycle(["#D55E00", "#0072B2", "#CC79A7", "#E69F00", "#56B4E9", "#009E73", "#F0E442"])

    num_topics = len(df_output_topic_word)
    run_id = df_output_topic_word['run_id'][0]

    if topics is not None:  # Get selected topics
        topics = sorted(topics)
    elif top_n_topics is not None and 0 < top_n_topics < num_topics:  # Get top n topics
        topics = sorted(df_output_topic_word['topic_num'].to_list())[:top_n_topics]
    else:  # Get all topics
        topics = sorted(df_output_topic_word['topic_num'].to_list())

    if algorithm == 'bertopic':
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
    row = 1
    column = 1
    for topic in topics:
        words = [word + "  " for word in topic_num_to_df[topic]['topic_words']][:n_words][::-1]
        scores = [score for score in topic_num_to_df[topic]['word_scores']][:n_words][::-1]

        # noinspection PyTypeChecker
        fig.add_trace(
            go.Bar(x=scores,
                   y=words,
                   orientation='h',
                   marker_color=next(colors)),
            row=row, col=column)

        if column == columns:
            column = 1
            row += 1
        else:
            column += 1

    # Stylize graph
    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        title={
            'text': f"<b>Topic Word Scores for algorithm=\"{algorithm}\" and run_id={run_id}",
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

    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    # fig = fig # type: plotly.graph_objs._figure.Figure
    # fig.write_image() # todo: add name of the score metric
    # todo: write as image
    # fig.write_image('asd.png')
    return fig
