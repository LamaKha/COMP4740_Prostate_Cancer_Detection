# NOTES
'''
ENSG00000275403.1 -> column with only zeros

'''
from data_loader import load_and_process_data
from classification import classify_and_validate
from feature_optimization import filter_features
from sklearn.model_selection import train_test_split


# MAIN PROGRAM ---------------------------------------------------------------

# Load and preprocess the data (data_loader.py)
# THIS FUNCTION SHOULD TAKE A FILE NAME AS A PARAMETER AND RETURN 
# THE SET OF FEATURES (X) AND TARGET VARIABLE (Y)
X, y = load_and_process_data('prad_tcga_genes.csv')



