import os
import glob

import pandas as pd
from sklearn.preprocessing import LabelEncoder

anomaly_cols = ['DDoS', 'PortScan', 'Bot', 'Infiltration',
       'Web Attack � Brute Force', 'Web Attack � XSS',
       'Web Attack � Sql Injection', 'FTP-Patator', 'SSH-Patator',
       'DoS slowloris', 'DoS Slowhttptest', 'DoS Hulk', 'DoS GoldenEye',
       'Heartbleed']

def create_dataset(data_path, n=None):
    """
    Args:
        data_path: path to dir with data files (*.csv)
        n: number of files to load

    Returns:
        dataset with cols Label_bin, Label_encoded(multilabel), Label_str
    """

    all_files = glob.glob(os.path.join(data_path, "*.csv"))
    print(f'found {len(all_files)} files')

    df_from_each_file = (pd.read_csv(f) for f in all_files[:n])
    df = pd.concat(df_from_each_file, ignore_index=True)

    print(f'number Na in df: {df.isna().sum().sum()}')
    df.dropna(inplace = True)

    anomaly_cols_map = {col:1 for col in anomaly_cols}
    anomaly_cols_map['BENIGN'] = 0

    df['Label_bin'] = df[' Label'].map(anomaly_cols_map)

    encoder = LabelEncoder()
    df['Label_encoded'] = encoder.fit_transform(df[' Label'])

    df['Label_str'] = df[' Label']
    df.drop([' Label'], axis=1, inplace=True)

    return df