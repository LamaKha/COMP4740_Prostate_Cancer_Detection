#Use this file to
#test all classifiers after initial feature selection
#and after additional feature selection
#it takes the logest amount of time to run (about 12 minutes)

import pandas as pd
from sklearn.svm import SVC
from collections import Counter
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB
from data_loader import load_and_process_data
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, SelectFromModel
import warnings
warnings.filterwarnings("ignore")

SEED = 42

# MAIN PROGRAM ---------------------------------------------------------------

# Load and preprocess the data (data_loader.py)
# THIS FUNCTION SHOULD TAKE A FILE NAME AS A PARAMETER AND RETURN
# THE SET OF FEATURES (X) AND TARGET VARIABLE (Y)
x, y = load_and_process_data('prad_tcga_genes.csv')

# Initial Feature Selection Methods--------------------------------------------

# Removal of Outliers
clf = IsolationForest(max_samples=100, random_state=SEED)
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

# For display only -----------------------------------------------------------
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(x.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Features','Score']  #naming the dataframe columns
print("==============================================================================================================")
print("Listing Selected Feature Scores")
print("==============================================================================================================")
print(featureScores.nlargest(10,'Score'))
# ----------------------------------------------------------------------------

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

#Additional Feature Selection Methods---------------------------------------------------------------------------------

#Feature Selection using tree based method

# NOT USED? --------------------------------------
clf = RandomForestClassifier(n_estimators=1)
clf = clf.fit(var_thres,y_no_outliers)

model = SelectFromModel(clf, prefit=True)
X_new_gleason = model.transform(var_thres)
# ------------------------------------------------

#Resampling dataset
sme = SMOTEENN(random_state=SEED, smote=SMOTE(random_state=SEED))
X_res_smoteenn, y_res_smoteenn = sme.fit_resample(x_no_outliers, y_no_outliers)
print('Resampling of dataset using SMOTEENN %s' % Counter(y_res_smoteenn), '\n')

lda = LinearDiscriminantAnalysis(n_components = 3)
X_lda = lda.fit_transform(X_res_smoteenn,y_res_smoteenn)
print("==============================================================================================================")

#Second Testing of Classifiers After adding more Feature Selection Methods--------------------------------------------
print("Testing of Classifiers After Applying Additional Feature Selection Methods")
#Naive Bayes on Resampled dataset
gnb = GaussianNB()
scores_nb=cross_val_score(gnb, X_lda,y_res_smoteenn, scoring='accuracy', cv=10)
print("Accuracy scores for Gaussian Naive Bayes algorithm on resampled dataset : \n" , scores_nb , '\n')
print("Mean Accuracy :" , scores_nb.mean())
print("==============================================================================================================")
# SVM on Resampled dataset
clf = SVC(gamma='auto')
scores_svm=cross_val_score(gnb, X_lda,y_res_smoteenn, scoring='accuracy', cv=10)
print("Accuracy scores for SVM-RBF algorithm on resampled dataset : \n" , scores_svm , '\n')
print("Mean Accuracy :" , scores_svm.mean())

print("==============================================================================================================")
#Random Forest on Resampled dataset
rf = RandomForestClassifier(n_estimators=100,n_jobs=-1,max_depth=10,max_features='auto')
scores_rf=cross_val_score(rf, X_lda,y_res_smoteenn, scoring='accuracy', cv=10)
print("Accuracy scores for Random Forest algorithm on resampled dataset : \n" , scores_rf , '\n')
print("Mean Accuracy :" , scores_rf.mean())
print("==============================================================================================================")
#KNN neighbours on resampled dataset
k_scores = []
# Calculating best values for K values between 1 and 20
for i in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=i)
    scores_knn = cross_val_score(knn, X_lda, y_res_smoteenn, cv=10, scoring='accuracy')
    k_scores.append(scores_knn.mean())
print("Accuracy scores for KNeighbors Classifier on resampled dataset : \n" , scores_knn , '\n')
print("Mean Accuracy :" ,scores_knn.mean())
print("==============================================================================================================")
