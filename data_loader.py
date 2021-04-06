import pandas as pd

def load_and_process_data(file_name):
    
    data = pd.read_csv(file_name)

    # Set index to ID values (otherwise keys are wrong after transposition)
    data = data.set_index('ID')
    
    # Transpose the data (because features are rows in CSV and they should be columns)
    data = data.transpose()
       
    # Remove the features that only contain 0 values
    data = data.loc[:, (data != 0).any(axis=0)]
    
    # Identify independant and dependant variables
    # Remove the last 6 columns
    X = data.iloc[:, : -6]
    # Temporarily setting target variable to 'GLEASON_PATTERN_PRIMARY'
    y = data.iloc[:, -5]
    
    return X, y