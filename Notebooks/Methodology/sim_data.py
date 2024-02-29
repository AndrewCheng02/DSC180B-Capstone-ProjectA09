import pandas as pd
import numpy as np
import pingouin as pgs
import os

import gzip # Extract gz file
from ast import literal_eval # Gets network matrix array from csv 
import re

def get_sim_data(partial = False, fischer = False):
    '''Gets the simulated data with the desired type of net work matrix transformation
    
    Args:
        partial -- Applys partial correlation
        fischer -- Applys fischer transformation
    
    Returns:
        List of distributions (Pandas Data Frame)
    
    '''

    distributions = []
    
    # Loop through all distributions
    for i in range(0, 11, 2):
        
        data_file = os.path.abspath('../../Data/sim/dist' + str(i) + '.csv.gz')
        
        try:
            with gzip.open(data_file) as filepath:

                # Network Matrix Parse function
                parse_net_mat = lambda x : np.array(literal_eval(re.sub('(?<!\[)\s+|[\\n]', ', ', x)))

                # Read the data from csv
                data = pd.read_csv(data_file, index_col = False, converters = {'corr' : parse_net_mat})

                if partial:
                    data['corr'] = data['corr'].apply(lambda x : x.pcorr())
                    
                if fischer:
                    data['corr'] = data['corr'].apply(lambda x : np.arctanh(x))

                distributions.append(data)
                
        except FileNotFoundError:
            print('File not found!')
        
    return distributions

