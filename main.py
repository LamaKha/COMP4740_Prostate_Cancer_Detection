# NOTES
'''
ENSG00000275403.1 -> column with only zeros

'''
from data_loader import load_data, preprocess_data
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


# MAIN PROGRAM ---------------------------------------------------------------

# Load the data from a CSV
data = load_data('prad_tcga_genes.csv')

# Set index to ID values (otherwise keys are wrong after transposition)
data = preprocess_data(data)


# Identify independant and dependant variables
# Remove the last 6 columns
X = data.iloc[:, : -6]
# Temporarily setting target variable to 'GLEASON_PATTERN_PRIMARY'
y = data.iloc[:, -5]

# Generate a train / test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 21)

# Apply Feature Selection Filter (chi2) to reduce the number of features
chi2_selector = SelectKBest(chi2, k=5000)
X_chi2_train = chi2_selector.fit_transform(X_train, y_train)
X_chi2_test = chi2_selector.transform(X_test)

# Fitting the Chi2 - Filtered Data Using A Random Forest
classifier = RandomForestClassifier(random_state = 42)
classifier.fit(X_chi2_train, y_train)

# Predict The Test Set Results
y_pred = classifier.predict(X_chi2_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

n = len(y_test)

print(cm)
print(n)
