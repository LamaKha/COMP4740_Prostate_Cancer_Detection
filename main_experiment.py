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
y_count = Counter(y)
print(y_count)
# Initial Feature Selection Methods--------------------------------------------


#selecting best k-features
bestfeatures = SelectKBest(score_func=chi2, k=15000)
fit = bestfeatures.fit(x, y)
best_x = fit.transform(x)

# Scaling the DataFrame
scaler = StandardScaler()
scaled_data = scaler.fit_transform(best_x)

# Eliminating low variance features
selector = VarianceThreshold()
var_thres = selector.fit_transform(scaled_data)
print(var_thres.shape)

#Additional Feature Selection Methods---------------------------------------------------------------------------------

#Feature Selection using tree based method
clf = RandomForestClassifier(n_estimators=100)
clf = clf.fit(var_thres,y)
model = SelectFromModel(clf, prefit=True)
X_new_gleason = model.transform(var_thres)
print(X_new_gleason.shape)

#Resampling dataset
sme = SMOTEENN(random_state=SEED, smote=SMOTE(random_state=SEED, k_neighbors=1))
X_res_smoteenn, y_res_smoteenn = sme.fit_resample(X_new_gleason, y)
print('Resampling of dataset using SMOTEENN %s' % Counter(y_res_smoteenn), '\n')

lda = LinearDiscriminantAnalysis(n_components = 3)
X_lda = lda.fit_transform(X_res_smoteenn,y_res_smoteenn)
print("==============================================================================================================")

#Second Testing of Classifiers After adding more Feature Selection Methods--------------------------------------------
print("Testing of Classifiers After Applying Additional Feature Selection Methods")

#Naive Bayes on Resampled dataset
gnb = GaussianNB()
scores_nb=cross_val_score(gnb, X_res_smoteenn,y_res_smoteenn, scoring='accuracy', cv=8)
print("Accuracy scores for Gaussian Naive Bayes algorithm on resampled dataset : \n" , scores_nb , '\n')
print("Mean Accuracy :" , scores_nb.mean())
print("==============================================================================================================")

# SVM on Resampled dataset
clf = SVC(gamma='auto')
scores_svm=cross_val_score(clf, X_res_smoteenn,y_res_smoteenn, scoring='accuracy', cv=8)
print("Accuracy scores for SVM-RBF algorithm on resampled dataset : \n" , scores_svm , '\n')
print("Mean Accuracy :" , scores_svm.mean())
print("==============================================================================================================")

#Random Forest on Resampled dataset
rf = RandomForestClassifier(n_estimators=100,n_jobs=-1,max_depth=10,max_features='auto')
scores_rf=cross_val_score(rf, X_res_smoteenn,y_res_smoteenn, scoring='accuracy', cv=8)
print("Accuracy scores for Random Forest algorithm on resampled dataset : \n" , scores_rf , '\n')
print("Mean Accuracy :" , scores_rf.mean())
print("==============================================================================================================")

#KNN neighbours on resampled dataset
k_scores = []
# Calculating best values for K values between 1 and 20
for i in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=i)
    scores_knn = cross_val_score(knn, X_res_smoteenn, y_res_smoteenn, cv=8, scoring='accuracy')
    k_scores.append(scores_knn.mean())
print("Mean Accuracy for each K belonging to  {1:21} : \n" , k_scores , '\n')
print("Best of the Mean Accuracies :" , max(k_scores))
print("==============================================================================================================")
