import os
import sys

# assert os.path.exists('../src'), f"[ERROR] The path src not detected in the parent directory '{os.getcwd()}'."
#
# if os.getcwd().endswith('/notebooks'):
#     os.chdir('..')
#     sys.path.append('./src')
sys.path.append('./src')
print(f'[INFO] Current Directory: "{os.getcwd()}".')

import ast
import pandas as pd
import time
import shutil

from src.bulk_runner import OUTPUT_FOLDER
from src.bulk_runner import main_runner

if os.path.exists(OUTPUT_FOLDER):
    shutil.rmtree(OUTPUT_FOLDER)

combinations = []

for comb_filename in ['combinations.txt', 'combinations_ctm.txt']:
    with open(comb_filename) as comb_file:
        for i, item in enumerate(comb_file.readlines()):
            x = ast.literal_eval(item)
            combinations.append(x)

dfs_merged = pd.DataFrame()
for i, configs in enumerate(combinations):
    print(f'[INFO] Configs #{i}/{len(combinations)}: {configs}')
    try:
        df_output_topic_word = main_runner(configs=configs)
        dfs_merged = pd.concat([dfs_merged, df_output_topic_word])
        dfs_merged.to_csv(f'./{OUTPUT_FOLDER}/merged.csv')
    except Exception:
        print('[WARN] Current execution gave an error!')
        time.sleep(10)
