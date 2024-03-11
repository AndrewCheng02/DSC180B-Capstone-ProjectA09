def uncompress(fpath):
    import pandas as pd
    import numpy as np
    import os

    import gzip # Extract gz file
    from ast import literal_eval # Gets network matrix array from csv 
    import re

    convert = lambda x : np.array(literal_eval(re.sub('(?<!\[)\s+|[\\n]', ', ', x)))

    with gzip.open(fpath) as filepath:
        return pd.read_csv(filepath, index_col = 'subject_id', converters = {'corr': convert})
