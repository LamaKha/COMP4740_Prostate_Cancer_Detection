import pipeline as pl

SEED = 42       # The random seed used
K_BEST = 15000  # The number of features to select using Chi2 filter
N_DIM = 3       # The number of dimensions (dimensionality reduction)
N_FOLDS = 8     # The number of folds to use in cross-fold validation
 
# Load and process the data and return the feature and target matrices -------
X, y = pl.load_and_process_data('prad_tcga_genes.csv', 'GLEASON_SCORE')

# Display the data profile ---------------------------------------------------
pl.data_profile(X, y, 'INITIAL')

# Perform feature selection --------------------------------------------------
X_select = pl.feature_selection(X, y, K_BEST)

# Perform up-sampling and down-sampling to address class imbalance -----------
X_balanced, y_balanced = pl.fix_imbalance(X_select, y, SEED)

# Display the updated data profile -------------------------------------------
pl.data_profile(X_balanced, y_balanced, 'IMBALANCE ADJUSTED')

# Perform dimensionality reduction (optional) --------------------------------
X_reduced = pl.reduce_dimensions(X_balanced, y_balanced, N_DIM)

# Perform classification -----------------------------------------------------
model_names = ['Naive Bayes', 'Support Vector Machine', 'Random Forest', 'K Nearest Neighbors']
n_models = len(model_names)
scores = []
averages = []
models = []
for i in range(n_models):
    m, s, a = pl.classify(X_balanced, y_balanced, model_names[i], N_FOLDS)
    models.append(m)
    scores.append(s)
    averages.append(a)
    
# Display results ------------------------------------------------------------
for i in range(n_models):
    pl.display_results(model_names[i], scores[i], averages[i])

# Perform classification after dimensionality reduction ----------------------
print("RESULTS AFTER DIMENSIONALITY REDUCTION")
model_names = ['Naive Bayes', 'Support Vector Machine', 'Random Forest', 'K Nearest Neighbors']
n_models = len(model_names)
scores = []
averages = []
models = []
for i in range(n_models):
    m, s, a = pl.classify(X_reduced, y_balanced, model_names[i], N_FOLDS)
    models.append(m)
    scores.append(s)
    averages.append(a)
    
# Display results ------------------------------------------------------------
for i in range(n_models):
    pl.display_results(model_names[i], scores[i], averages[i])

    
    
