import pandas as pd
import sys
import os
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.getenv('PROJ_DIR', ''))

from data_loader.load_deepline import ALL_DATASET_NAMES

records = pd.read_csv('results/benchmark_results600/experiment_records.csv')

group_counts = records.groupby('dataset').size()

print(group_counts)
print(len(group_counts))

