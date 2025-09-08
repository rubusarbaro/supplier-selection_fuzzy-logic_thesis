import numpy as np
import pandas as pd

def mean_not_outliers(df: pd.DataFrame, column: str, filter_mask: pd.Series) -> float:
    values = df[filter_mask][column]

    mean_0 = values.mean()
    std_0 = values.std()

    lower = mean_0 - 3 * std_0
    upper = mean_0 + 3 * std_0

    non_outliers_mask = filter_mask & (df[column] >= lower) & (df[column] <= upper)

    mean = df[non_outliers_mask][column].mean()

    if np.isnan(mean):
        mean = mean_0

    return mean