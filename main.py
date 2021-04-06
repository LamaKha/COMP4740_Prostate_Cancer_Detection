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

# Generate a train / test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 21)

# Apply Feature Selection Filters (feature_optimization.py)
# THIS FUNCTION SHOULD TAKE THE FEATURES FROM THE TRAINING AND TEST SET AND
# RETURN FEATURE OPTIMIZED VERSIONS OF THESE ARRAYS
X_train, X_test = filter_features(X_train, y_train, X_test, 'chi2', 5000)

# Classify & Validate
# THIS FUNCTION SHOULD TAKE THE TRAINING AND TESTS SETS, CONDUCT CLASSIFICATION,
# AND THEN VALIDATE THE RESULTS
classify_and_validate(X_train, y_train, X_test, y_test, 'RF')

