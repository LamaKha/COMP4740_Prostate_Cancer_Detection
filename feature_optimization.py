from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

def filter_features(X_train, y_train, X_test, filter, n):
    if (filter == 'chi2'):
        selector = SelectKBest(chi2, k = n)
    if  (filter == 'VT'):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(SelectKBest(chi2, k = n))
        selector = VarianceThreshold(scaled_data)
    if  (filter == 'RF'):
        selector = SelectFromModel(RandomForestClassifier(n_estimators=n))

    return selector.fit_transform(X_train, y_train), selector.transform(X_test)

def feature_search(X, y):
    return
    
