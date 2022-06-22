import json
import typing
from pathlib import Path
from typing import List, Tuple

import pandas as pd


def load_documents(dataset_dir: str, dataset_text_col: str = None) -> Tuple[List[str], List[str]]:
    if '20news_bydate' in dataset_dir:
        dataset_data_path = [path for path in Path(dataset_dir).iterdir() if path.suffix == '.csv'][0]
        df = pd.read_csv(dataset_data_path)
        documents = list(map(lambda doc: '' if pd.isna(doc) else doc, df['text']))  # Replace nan with ''
        labels = list(map(lambda doc: '' if pd.isna(doc) else doc, df['label']))  # Replace nan with ''
        return documents, labels

    dataset_data_paths = sorted([path for path in Path(dataset_dir).iterdir() if path.suffix in {'.csv', '.tsv'}])

    labels_as_filenames = []
    dfs = []
    for data_path in dataset_data_paths:
        csv_delimiter = '\t' if data_path.suffix == '.tsv' else ','
        df = pd.read_csv(data_path, delimiter=csv_delimiter)
        labels_as_filenames.extend([Path(data_path).stem] * len(df))
        # print(f'[INFO] Dataset from "{data_path}":', tabulate(df.head(5), headers="keys", tablefmt="psql"), sep='\n')
        dfs.append(df)

    merged_df = pd.concat(dfs, axis=0)
    documents = list(map(lambda doc: '' if pd.isna(doc) else doc, merged_df[dataset_text_col]))  # Replace nan with ''
    return documents, labels_as_filenames


def pretty_print_dict(a_dict: typing.Dict, indent=4, info_log: str = None):
    if info_log:
        print('[INFO]', info_log)
    print(json.dumps(a_dict, indent=indent))
