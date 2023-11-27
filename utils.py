import numpy as np


def replace_outliers_IQR(column, multiplier=1.5, strategy='clamp', replace_with=None):
    q1 = column.quantile(0.25)
    q3 = column.quantile(0.75)
    iqr = q3 - q1
    low_lim = q1 - multiplier * iqr
    high_lim = q3 + multiplier * iqr

    if strategy == 'clamp':
        column = np.where(column >= high_lim, high_lim, np.where(column <= low_lim, low_lim, column))
    elif strategy == 'replace':
        if replace_with is not None:
            column = np.where(column < low_lim, replace_with, column)
            column = np.where(column > high_lim, replace_with, column)
        else:
            raise ValueError("replace_with parameter must be specified when strategy is 'replace'")
    else:
        raise ValueError("Invalid strategy. Supported strategies are 'clamp' and 'replace'.")

    return column