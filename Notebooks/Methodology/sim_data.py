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
    
    def pcorr(X_corr):
        X_inv = np.linalg.inv(X_corr)
        X_pcorr = -X_inv / np.outer(np.sqrt(np.diag(X_inv)), np.sqrt(np.diag(X_inv)))
        
        return X_pcorr

    distributions = []
    
    # Loop through all distributions
    for i in range(1, 11):
        
        data_file = os.path.abspath('../../Data/sim/dist' + str(i).rjust(2, '0') + '.csv.gz')
        
        try:
            with gzip.open(data_file) as filepath:

                # Network Matrix Parse function
                parse_net_mat = lambda x : np.array(literal_eval(re.sub('(?<!\[)\s+|[\\n]', ', ', x)))

                # Read the data from csv
                data = pd.read_csv(data_file, index_col = False, converters = {'corr' : parse_net_mat})

                if partial:
                    data['corr'] = data['corr'].apply(lambda x : pcorr(x))
                    
                if fischer:
                    data['corr'] = data['corr'].apply(lambda x : np.arctanh(x))

                distributions.append(data)
                
        except FileNotFoundError:
            print('File not found!')
        
    return distributions

