import numpy as np
import pandas as pd
from utils import * 
import pytest


@pytest.mark.parametrize("column, strategy, replace_with, expected",
                        [(pd.Series([1, 2, 3, 4, 5, 100]), 'clamp', None, pd.Series([1, 2, 3, 4, 5, 8.5])),
                         pytest.param(pd.Series([1, 2, 3, 4, 5, 100]), 'clamp', None, pd.Series([1, 2, 3, 4, 5, 8.5]), marks=pytest.mark.xfail), # expected pass
                        (pd.Series([1, 2, 3, 4, 5, 100]), 'replace', 10, pd.Series([1, 2, 3, 4, 5, 10])), 
                        pytest.param(pd.Series([1, 2, 3, 4, 5, 100]), 'replace', 10, pd.Series([1, 2, 3, 4, 5, 10]), marks=pytest.mark.xfail), # expected pass
                        (pd.Series([1, 2, 3, 4, 5, 20, 30, 100]), 'replace', None, ValueError),
                        (pd.Series([1, 2, 3, 4, 5, 20, 30, 100]), 'delete', None, ValueError)]) 

def test_replace_strategy(column, strategy, replace_with, expected):
    if strategy not in ['clamp', 'replace']:
        with pytest.raises(ValueError, match="Invalid strategy. Supported strategies are 'clamp' and 'replace'."):
            replace_outliers_IQR(column, strategy=strategy, replace_with=replace_with)
    elif isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(ValueError, match="replace_with parameter must be specified when strategy is 'replace'"):
            replace_outliers_IQR(column, strategy=strategy, replace_with=replace_with)
    else:
        observed = replace_outliers_IQR(column, strategy=strategy, replace_with=replace_with)
        np.testing.assert_allclose(expected, observed)
