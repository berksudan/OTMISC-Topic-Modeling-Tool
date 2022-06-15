import pickle as pkl
import typing
import zlib
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple
import json
import pandas as pd


def load_documents(dataset_dir: str, dataset_text_col: str) -> Tuple[List[str], List[str]]:
    if '20news_bydate' in dataset_dir:
        dataset_data_path = [path for path in Path(dataset_dir).iterdir() if path.suffix == '.pkz'][0]
        decompressed_pkl = zlib.decompress(open(dataset_data_path, 'rb').read())

        data = pkl.loads(decompressed_pkl)
        return data['train'].data

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


