import numpy as np
import pandas as pd

def clean_dataset(df, to_float64=False):
    """
    Args:
        df: pd.DataFrame
        to_float64: variable type change for pytorch
    Returns:
        drop nan, inf values from DataFrame
    """
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.select_dtypes(include=['float64', 'float32', 'float16', 'int64', 'int32', 'int16', 'int8'])\
        .isin([np.nan, np.inf, -np.inf]).any(1)
    if to_float64:
        mask = df.select_dtypes(include=['float64', 'float32', 'float16', 'int64', 'int32', 'int16', 'int8'])
        df[mask] = df[mask].astype(np.float64)
    return df[indices_to_keep]