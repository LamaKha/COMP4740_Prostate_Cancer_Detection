import pandas as pd

def load_data(name):
    
    # Load the data from a CSV
    return pd.read_csv('prad_tcga_genes.csv')


def preprocess_data(data):
    # Set index to ID values (otherwise keys are wrong after transposition)
    data = data.set_index('ID')
    
    # Transpose the data (because features are rows in CSV and they should be columns)
    data = data.transpose()
       
    # Remove the features that only contain 0 values
    data = data.loc[:, (data != 0).any(axis=0)]
    
    return data