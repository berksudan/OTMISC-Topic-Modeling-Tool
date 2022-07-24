import os
import shutil
from time import time, sleep
from typing import Dict

import pandas as pd

from src import evaluator
from src import preprocessor
from src import visualizer
from utils import load_documents

OUTPUT_FOLDER = './output'
AVAILABLE_ALGORITHMS = {'lda', 'nmf', 'ctm', 'lda-bert', 'bertopic', 'top2vec'}


def main_runner(configs: Dict):
    algorithm_args = configs['algorithm_args']
    algorithm_name = algorithm_args['algorithm'].lower()
    run_id = int(time())

    docs, labels = load_documents(dataset=configs['dataset'])
    if 'preprocessing_funcs' in configs:
        docs = preprocessor.run(data=docs, prep_functions=configs['preprocessing_funcs'])

    algorithm_args.update(data_name=configs['dataset'], docs=docs, labels=labels)
    print(f'[INFO] Running with {algorithm_args["num_topics"]} topics.')

    output_folder = f'{OUTPUT_FOLDER}/{run_id}_{algorithm_name}'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if algorithm_name in {'bertopic', 'lda-bert'}:
        from src.bertopic_runner import BertopicTrainer

        if algorithm_name == 'lda-bert':
            print('[WARN] Lda-Bert is experimental and use with caution!')
        trainer = BertopicTrainer(dataset=configs['dataset'], model_name=algorithm_name, args=algorithm_args,
                                  run_id=run_id)

        model, df_output_doc_topic, df_output_topic_word = trainer.train()
    elif algorithm_name in {'lda', 'nmf', 'ctm'}:
        from src import LDA_NMF_CTM_runner
        if algorithm_name == 'ctm':
            print('[WARN] CTM is experimental and does not guarantee reproducibility. Please use with caution!')
        df_output_doc_topic, df_output_topic_word = LDA_NMF_CTM_runner.runner(
            args=algorithm_args, run_id=run_id, output_folder=output_folder, model_name=algorithm_name)
        model = None
    elif algorithm_name == 'top2vec':
        from src import top2vec_runner
        algorithm_args.update(run_id=run_id)
        model, df_output_doc_topic, df_output_topic_word = top2vec_runner.parametric_run(args=algorithm_args)
    else:
        raise ValueError(f'Algorithm {algorithm_name} is not available, available algorithms:{AVAILABLE_ALGORITHMS}.')

    df_output_topic_word = evaluator.compute_topic_scores(df_output_doc_topic, df_output_topic_word)
    try:
        df_output_doc_topic.to_csv(f'{output_folder}/output_doc_topic.tsv', sep='\t', index=False)
        df_output_topic_word.to_csv(f'{output_folder}/output_topic_word.tsv', sep='\t', index=False)

        # if model:
        #     visualizer.visualize_topic_similarity_matrix(model, df_output_doc_topic, df_output_topic_word,
        #                                                  target_dir=output_folder)
        #     print('[INFO] Created Top Words Barchart Visualization successfully.')
        #     visualizer.draw_umap2d_scatter_plot(model, df_output_topic_word, df_output_doc_topic,
        #                                         target_dir=output_folder)
        # visualizer.visualize_labels_per_topic(df_output_doc_topic, df_output_topic_word, top_n_topics=10,
        #                                       target_dir=output_folder)
        # visualizer.visualize_top_words_barchart(df_output_topic_word, n_words=5, target_dir=output_folder)
        # visualizer.draw_representative_docs(df_output_doc_topic, top_n_docs=3)
    except:
        pass
    return df_output_topic_word


def main():
    from configs_list import configs_list
    if os.path.exists(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER)
    dfs_output_topic_word = []
    for configs in configs_list:
        with open('file.txt', 'r') as fin:
            data = fin.read().splitlines(True)
        with open('file.txt', 'w') as fout:
            fout.writelines(data[1:])
        try:
            df_output_topic_word = main_runner(configs=configs)
            dfs_output_topic_word.append(df_output_topic_word)
        except Exception:
            print('[WARN] Current execution gave an error!')
            sleep(10)
            continue
    pd.concat(dfs_output_topic_word).to_csv(f'./{OUTPUT_FOLDER}/merged.csv')


if __name__ == '__main__':
    main()
