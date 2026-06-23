import sys
import datetime
import os
import traceback
import pandas as pd
import json
import numpy as np
import random
import time

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder


from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.getenv('PROJ_DIR', ''))

from parallel_evolutionary_pipeline.datasets import OpenMLDataset, _normalize_frame

# /path/to/deepline_dataset/[dataset_name]/data.csv
# /path/to/deepline_dataset/[dataset_name]/info.json

BASE_DIR = os.getenv('PREPBENCH_DATA_DIR', os.path.join(
    os.path.dirname(__file__),
    '..', '..',
    'data/deepline_dataset'
))
ALL_DATASET_NAMES = os.listdir(BASE_DIR)


def load_deepline_Xy(name: str):
    if name not in ALL_DATASET_NAMES:
        raise ValueError(f'`{name}` not in available datasets!')
    
    df = pd.read_csv(os.path.join(BASE_DIR, name, 'data.csv'))
    with open(os.path.join(BASE_DIR, name, 'info.json'), 'r') as f:
        info = json.load(f)
    label_name: str = info['label']
    X = df.drop(columns=[label_name])
    y = df[label_name]
    X, numeric_columns, categorical_columns = _normalize_frame(X)
    y = y.astype("category")
    return X, y, numeric_columns, categorical_columns

def load_deepline_dataset(name: str) -> OpenMLDataset:
    X, y, numeric_columns, categorical_columns = load_deepline_Xy(name)
    
    return OpenMLDataset(
        name=name,
        X=X.reset_index(drop=True),
        y=y.reset_index(drop=True),
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )
