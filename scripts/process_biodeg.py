import pandas_flavor as pf
from sklearn.preprocessing import scale, StandardScaler
from tqdm import tqdm

@pf.register_dataframe_method
def standardize(df, columns=None):
    """
    Standardizes a column to the prescribed mean and standard deviation.
    
    Defaults to standard scaling.
    """
    if not columns:
        columns = list(df.columns)
    for column in tqdm(columns):
        mean = df[column].mean()
        std = df[column].std()
        df.loc[:, column] = df.loc[:, column].apply(lambda x: (x - mean) / std)
    return df


import pandas as pd
import yaml
import janitor

with open('../data/biodeg.yaml', 'r+') as f:
    columns = yaml.load(f)

data = (
    pd.read_csv('../data/biodeg.csv', sep=';', header=None)
    .rename(mapper=dict(zip(range(len(columns)), columns.keys())), axis=1)
    .clean_names()
    .label_encode('experimental_class')
    .remove_columns(['experimental_class'])
    .rename_column('experimental_class_enc', 'experimental_class')
)
X, y = data.get_features_targets(target_columns=['experimental_class'])
X = X.standardize()

X.to_csv('../data/biodeg_X.csv')
y.to_csv('../data/biodeg_y.csv')