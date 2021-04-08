#Use this file to
#test all classifiers after initial feature selection ONLY
#It takes less than full testing but longer than second testing (about 10 minutes)

import pandas as pd
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from data_loader import load_and_process_data
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
import warnings
warnings.filterwarnings("ignore")

# MAIN PROGRAM ---------------------------------------------------------------

# Load and preprocess the data (data_loader.py)
# THIS FUNCTION SHOULD TAKE A FILE NAME AS A PARAMETER AND RETURN
# THE SET OF FEATURES (X) AND TARGET VARIABLE (Y)
x, y = load_and_process_data('prad_tcga_genes.csv')

# Initial Feature Selection Methods--------------------------------------------

# Removal of Outliers
clf = IsolationForest(max_samples=100, random_state=None)
clf.fit(x)

# predictions
outliers = clf.predict(x)
outliers_series = pd.Series(outliers)
x['outliers'] = outliers_series.values
x['gleason_score'] = y.values
x_no_outliers = x[x.outliers != -1]
y_no_outliers = x_no_outliers['gleason_score']
x_no_outliers.drop(['outliers','gleason_score'],axis=1,inplace=True)

#selecting best k-features
bestfeatures = SelectKBest(score_func=chi2, k=15000)
fit = bestfeatures.fit(x_no_outliers,y_no_outliers)
best_x = fit.transform(x_no_outliers)
best_x.shape
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(x.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Features','Score']  #naming the dataframe columns
print("==============================================================================================================")
print("Listing Selected Feature Scores")
print("==============================================================================================================")
print(featureScores.nlargest(10,'Score'))

# Scaling the DataFrame
scaler = StandardScaler()
scaled_data = scaler.fit_transform(best_x)

# Eliminating low variance features
selector = VarianceThreshold()
var_thres = selector.fit_transform(scaled_data)

#Initial Testing of Classifiers--------------------------------------------------------------------------------------
print("==============================================================================================================")
print("Testing of Classifiers After Applying Initial Feature Selection Methods")
print("==============================================================================================================")
#SVM
clf = SVC(gamma='auto')
scores_svm = cross_val_score(clf, x, y, scoring='accuracy', cv=10)
print("Accuracy scores for SVM-RBF algorithm on Original dataset : \n", scores_svm, '\n')
print("Mean Accuracy :", scores_svm.mean(), '\n')
print("==============================================================================================================")
#Naive Bayes
gnb = GaussianNB()
scores = cross_val_score(gnb, x, y, scoring='accuracy', cv=10)
print("Accuracy scores for Gaussian Naive Bayes algorithm on Original dataset : \n", scores, '\n')
print("Mean Accuracy :", scores.mean(), '\n')
print("==============================================================================================================")
# Random Forest on Original Dataset
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, max_depth=10, max_features='auto')
scores = cross_val_score(rf, x, y, scoring='accuracy', cv=10)
print("Accuracy scores for Random Forest algorithm on Original dataset : \n", scores, '\n')
print("Mean Accuracy :", scores.mean(), '\n')
print("==============================================================================================================")
# KNN neighbours on original dataset
k_scores = []
# Calculating best values for K values between 1 and 20
for i in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=i)
    scores = cross_val_score(knn, x, y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
print("Accuracy scores for K-NN algorithm on Original dataset : \n", scores, '\n')
print(k_scores)
print("Mean Accuracy :", scores.mean(), '\n')
print("==============================================================================================================")
