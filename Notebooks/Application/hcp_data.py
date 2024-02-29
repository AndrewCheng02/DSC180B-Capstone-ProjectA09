import pandas as pd
import numpy as np
import os

import gzip # Extract gz file
from ast import literal_eval # Gets network matrix array from csv 
import re

data_file = os.path.abspath('../../Data/data_clean.csv.gz')

def get_hcp_cleaned_data():
    with gzip.open(data_file) as filepath:
        return pd.read_csv(filepath, index_col = 'subject_id', 
            converters = {'netmat' : lambda x : np.array(literal_eval(re.sub('(?<!\[)\s+|[\\n]', ', ', x)))})
