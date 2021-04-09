import pandas as pd
from collections import Counter
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, SelectFromModel
from sklearn.metrics import confusion_matrix


def load_and_process_data(file_name, target):
    """
    load_and_process_data reads a csv from the working directory, 
    transposes the resulting dataframe (so the rows of the original CSV 
    become the features), removes any features that contain 0 values, adds
    GRADE_GROUP as a dependant variable (if desired) and returns the 
    feature matrix and target variable matrix (which can be set by the user).

    :param file_name: the name of the file ['example.csv'] (string)
    :param target: the name of the row containing the dependant variable (string)
    :return: X [the feature matrix] and y [the target matrix] (both numpy arrays)
    """ 
    data = pd.read_csv(file_name)

    # Set index to ID values (otherwise keys are wrong after transposition)
    data = data.set_index('ID')

    # Transpose the data (because features are rows in CSV and they should be columns)
    data = data.transpose()

    # Remove the features that only contain 0 values
    data = data.loc[:, (data != 0).any(axis=0)]

    # Add new target variable (GRADE_GROUP)
    if (target == 'GRADE_GROUP'):
        targets = []
        for index, row in data.iterrows():
            # GLEASON SCORE = 6 -> GRADE GROUP = 1
            if (row['GLEASON_SCORE'] == 6):
                targets.append(1)
            # GLEASON SCORE = 7 -> GRADE GROUP = 2/3
            elif (row['GLEASON_SCORE'] == 7):
                # PRIMARY SCORE = 3 -> GRADE GROUP = 2
                if (row['GLEASON_PATTERN_PRIMARY'] == 3):
                    targets.append(2)
                # PRIMARY SCORE = 4 -> GRADE GROUP = 3
                else:
                    targets.append(3)
            # GLEASON SCORE = 8 -> GRADE GROUP = 4
            elif (row['GLEASON_SCORE'] == 8):
                targets.append(4)
            # GLEASON SCORE = 9/10 -> GRADE GROUP = 5
            else:
                targets.append(5)
        data.insert(len(data.columns), "GRADE_GROUP", targets)
        # Remove the last 6 columns from the feature matrix
        X = data.iloc[:, : -6]
    else:
        # Remove the last 5 columns from the feature matrix
        X = data.iloc[:, : -5]        

    # Set the target variable matrix
    y = data[target]

    return X, y

def print_divider():
    """
    print_divider prints a dividing line (===).
    """ 
    print("==============================================================================================================")


def data_profile(X, y, phase):
    """
    data_profile displays key components of the data to the user.

    :param X: the feature  (numpy array)
    :param y: the target matrix (numpy array)
    :param phase: the current phase of data processing (string) 
    :return: None
    """ 
    print_divider()
    print("DATA PROFILE - ", phase)
    print("------------")
    tally = Counter(y)
    print("Number of samples: ", X.shape[0])
    print("Number of features: ", X.shape[1])
    print("Class Distribution:")
    print(tally)
    print_divider()   
    
def feature_selection(X, y, n):
    """
    feature_selection performs a series of feature selection steps on the 
    data. First the features are passed through a Chi2 filter and the 'n'
    best features are selected. Then the features are transformed using a 
    standard scaler (normalized). Then 0 variance features are removed. Lastly, 
    another round of feature selection is performed, using a random forest.

    :param X: the feature matrix (numpy array)
    :param y: the target matrix (numpy array)
    :param n: the number of features to select with Chi2 filter (int)
    :return: X_select - the feature matrix (numpy array)
    """ 
    # Apply Chi2 filter to reduce eliminate unimportant features
    bestfeatures = SelectKBest(score_func=chi2, k=n)
    fit = bestfeatures.fit(X, y)
    X_filtered = fit.transform(X)
    plot_best_scores(fit, X)
    
    # Normalize the remaining features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filtered)

    # Eliminate features that do not various accross the population
    selector = VarianceThreshold()
    X_threshold = selector.fit_transform(X_scaled)
    
    # Apply 2nd round of feature selection using a random forest
    clf = RandomForestClassifier(n_estimators=100)
    clf = clf.fit(X_threshold, y)
    model = SelectFromModel(clf, prefit=True)
    X_select = model.transform(X_threshold)
    
    return X_select

def fix_imbalance(X, y, seed):
    """
    fix_imbalance applies a Synthetic Minority Over-sampling Technique (SMOTE)
    to upsample the minority classes. It then uses an Edited Nearest Neibor
    (ENN) technique to downsample over-represented classes.

    :param X: the feature matrix (numpy array)
    :param y: the target matrix (numpy array)
    :param seed: the value used to seed the random number generator (int)
    :return: X_smoteen, y_smoteen - the augmented feature and target matrices (numpy arrays)
    """
    sme = SMOTEENN(random_state=seed, smote=SMOTE(random_state=seed, k_neighbors=1))
    X_smoteen, y_smoteen = sme.fit_resample(X, y)
    
    return X_smoteen, y_smoteen    
    
def reduce_dimensions(X, y, n_dim):
    """
    reduce_dimensions performs dimensionality reduction by applying 
    linear discriminant analysis. This can boost classification results 
    (although we lose the connection between the features and the result).
    :param X: the feature matrix (numpy array)
    :param y: the target matrix (numpy array)
    :param n_dim: the number of features after transformation (int)
    :return: X_lda - the transformed feature matrix (numpy array)
    """
    lda = LinearDiscriminantAnalysis(n_components = n_dim)
    X_lda = lda.fit_transform(X, y)
    
    return X_lda

def classify(X, y, classifier, n_folds, hyperparameters = None):
    """
    classify classifies the data using one of four classifiers: Naive Bayes, 
    Support Vector Machine, Random Forest, or K Nearest Neighbor.

    :param X: the feature matrix (numpy array)
    :param y: the target matrix (numpy array)
    :param classifier: the name of the classifier to be used (string)
    :param n_folds: the number of folds to use in cross validation
    :return: model (an untrained classification model)
             scores - a list of accuracies produced through cross validation (float list)
             average - the average accuracy for all rounds of cross validation (float)
             cm - the confusion matrix
    """
    # Naive Bayes Classifier
    if (classifier == 'Naive Bayes'):
        model = GaussianNB()
    # Support Vector Machine Classifier:
    if (classifier == 'Support Vector Machine'):
        model = SVC()    
    # Random Forest Classifier
    if (classifier == 'Random Forest'):
        model = RandomForestClassifier(n_jobs = -1, max_depth = 10)
    # K Nearest Neighbors Classifier
    if (classifier == 'K Nearest Neighbors'):
        k = find_best_k(X, y, n_folds)
        model = KNeighborsClassifier(n_neighbors = k)
    # Perform cross-validation and obtain set of 10 accuracy values
    scores = cross_val_score(model, X, y, scoring = 'accuracy', cv = n_folds)
    # Find the average accuracy
    average = scores.mean()
    
    # Obtain the confusion matrix
    y_pred = cross_val_predict(model, X, y, cv = n_folds)
    cm = confusion_matrix(y, y_pred)   
    metrics = analyze_cm(cm, len(y))
    
    return model, scores, average, cm, metrics
    
def find_best_k(X, y, k):  
    """
    find_best_k takes an array of feature data from a dataset (in this case 
    there are two features, x1 and x2), an array of corresponding lables identifying 
    each datapoint (as either belonging to class 0 or class 1 in this case) 
    and the number of folds (used in k-fold cross validation). It then runs a k-NN
    classifier using values ranging from 1-20 for 'k' and returns the value of 'k' 
    that achieved the best results (with respect to accuracy).

    :param X: the feature matrix
    :param y: the labels vector
    :param k: the number of folds used in k-fold cross validation
    :return: void (output is sent to the display)
    """ 
    kf = KFold(n_splits=k)
    
    range_k = range(1, 20)
    scores_list = []
    for k in range_k:
        model = KNeighborsClassifier(n_neighbors=k)
        score = cross_val_score(model, X, y, cv = kf)
        scores_list.append(score.mean())
    
    # OPTIONAL PLOTTING
    #plt.plot(range_k, scores_list)
    #plt.xlabel("Value of K")
    #plt.ylabel("Accuracy")
    #plt.show()
    
    max_accuracy = max(scores_list)
    best_k = scores_list.index(max_accuracy) + 1
    
    return best_k

def display_results(model_name, scores, average):
    """
    display_results displays the results obtained via classification of the 
    data: the model applied, the accuracies achieved through each fold of
    cross validation, and the average accuracy.

    :param model_name: the model used for classification (string)
    :param scores: the accuracies achieved through cross validation (float array)
    :param average: the average accuracy (float)
    :return: None
    """
    print_divider()
    print("Model: ", model_name)
    print("Accuracy Scores: ")
    print(scores)
    print("Average accuracy: ", round(average, 5))
    print_divider()
  
def plot_best_scores(model, X):
    """
    plot_best_scores plots the 10 features that obtained the highest scores
    via Chi2 selection on a bar chart.

    :param model: the fit model (object)
    :param X: a Dataframe containing the scores and feature names (Dataframe)
    :return: None
    """
    dfscores = pd.DataFrame(model.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Features','Score']  #naming the dataframe columns

    # OPTIONAL PLOTTING
    scores_array = featureScores.nlargest(10,'Score')
    scores_list = scores_array['Score'].tolist()
    features_list = scores_array['Features'].tolist()
    plt.bar(features_list, scores_list)
    plt.title("Most Relevant Features by Chi2 Score", fontsize = 18)
    plt.xlabel("Exon ID", fontsize = 16)
    plt.ylabel("Chi2 Score", fontsize = 16)
    plt.xticks(rotation=90)
    plt.show()

def analyze_cm(cm, n):
    """
    analyze_cm takes a confusion matrix as input and returns several
    measures: accuracy, sensitivity, specificity, precision, and NPV.

    :param cm: a confusion matrix (numpy array)
    :param n: The total number of samples (int)
    :return: A dictionary containing the metrics
    """
    # Expected VS Actual Probabilities
    pBB = cm[0, 0]
    pHB = cm[0, 1]
    pLB = cm[0, 2]
    pNB = cm[0, 3]
    pZB = cm[0, 4]
    
    pBH = cm[1, 0]
    pHH = cm[1, 1]
    pLH = cm[1, 2]
    pNH = cm[1, 3]
    pZH = cm[1, 4]
    
    pBL = cm[2, 0]
    pHL = cm[2, 1]
    pLL = cm[2, 2]
    pNL = cm[2, 3]
    pZL = cm[2, 4]
    
    pBN = cm[3, 0]
    pHN = cm[3, 1]
    pLN = cm[3, 2]
    pNN = cm[3, 3]
    pZN = cm[3, 4]
    
    pBZ = cm[4, 0]
    pHZ = cm[4, 1]
    pLZ = cm[4, 2]
    pNZ = cm[4, 3]
    pZZ = cm[4, 4]
    
    # True Positives
    tpB = pBB
    tpH = pHH
    tpL = pLL
    tpN = pNN
    tpZ = pZZ
    
    tp = tpB + tpH + tpL + tpN + tpZ
    
    # True Negatives
    tnB = pHH + pLH + pNH + pZH + pHL + pLL + pNL + pZL + pHN + pLN + pNN + pZN + pHZ + pLZ + pNZ + pZZ
    tnH = pBB + pLB + pNB + pZB + pBL + pLL + pNL + pZL + pBN + pLN + pNN + pZN + pBZ + pLZ + pNZ + pZZ
    tnL = pBB + pHB + pNB + pZB + pBH + pHH + pNH + pZH + pBN + pHN + pNN + pZN + pBZ + pHZ + pNZ + pZZ
    tnN = pBB + pHB + pLB + pZB + pBH + pHH + pLH + pZH + pBL + pHL + pLL + pZL + pBZ + pHZ + pLZ + pZZ
    tnZ = pBB + pHB + pLB + pNB + pBH + pHH + pLH + pNH + pBL + pHL + pLL + pNL + pBN + pHN + pLN + pNN
    
    # False Positives
    fpB = pBH + pBL + pBN + pBZ
    fpH = pHB + pHL + pHN + pHZ
    fpL = pLB + pLH + pLN + pLZ
    fpN = pNB + pNH + pNL + pNZ
    fpZ = pZB + pZH + pZL + pZN
    
    # False Negatives
    fnB = pHB + pLB + pNB + pZB
    fnH = pBH + pLH + pNH + pZH
    fnL = pBL + pHL + pNL + pZL
    fnN = pBN + pHN + pLN + pZN
    fnZ = pBZ + pHZ + pLZ + pNZ
    
    # TPR (Sensitivity) -> TP / (TP + FN)
    sensitivity = {}
    sensitivity['B'] = tpB / (tpB + fnB)
    sensitivity['H'] = tpH / (tpH + fnH)
    sensitivity['L'] = tpL / (tpL + fnL)
    sensitivity['N'] = tpN / (tpN + fnN)
    sensitivity['Z'] = tpZ / (tpZ + fnZ)
    
    average_sensitivity = 0
    for k, v in sensitivity.items():
        average_sensitivity = average_sensitivity + v
    average_sensitivity = average_sensitivity / len(sensitivity) 
    
    # TNR (Specificity) -> TN / (TN + FP)
    specificity = {}
    specificity['B'] = tnB / (tnB + fnB)
    specificity['H'] = tnH / (tnH + fnH)
    specificity['L'] = tnL / (tnL + fnL)
    specificity['N'] = tnN / (tnN + fnN)
    specificity['Z'] = tnZ / (tnZ + fnZ)
    
    average_specificity = 0
    for k, v in specificity.items():
        average_specificity = average_specificity + v
    average_specificity = average_specificity / len(specificity) 
    
    # PPV (Precision) -> TP / (TP + FP)
    precision = {}
    precision['B'] = tpB / (tpB + fpB)
    precision['H'] = tpH / (tpH + fpH)
    precision['L'] = tpL / (tpL + fpL)
    precision['N'] = tpN / (tpN + fpN)
    precision['Z'] = tpZ / (tpZ + fpZ)
    
    average_precision = 0
    for k, v in precision.items():
        average_precision = average_precision + v
    average_precision = average_precision / len(precision) 
    
    # NPV -> TN / (TN + FN)
    npv = {}
    npv['B'] = tnB / (tnB + fnB)
    npv['H'] = tnH / (tnH + fnH)
    npv['L'] = tnL / (tnL + fnL)
    npv['N'] = tnN / (tnN + fnN)
    npv['Z'] = tnZ / (tnZ + fnZ) 
    
    average_npv = 0
    for k, v in npv.items():
        average_npv = average_npv + v
    average_npv = average_npv / len(npv)
    
    # Accuracy
    accuracy = tp / n
    
    metrics = {'Accuracy': round(accuracy, 4),
               'Sensitivity': round(average_sensitivity, 4), 
               'Specificity': round(average_specificity, 4), 
               'Precision': round(average_precision, 4), 
               'NPV': round(average_npv, 4)}

    return metrics
    
    
    
    
    
    