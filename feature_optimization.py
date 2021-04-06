from sklearn.feature_selection import SelectKBest, chi2


def filter_features(X_train, y_train, X_test, filter, n):
    if (filter == 'chi2'):
        selector = SelectKBest(chi2, k = n)
    
    return selector.fit_transform(X_train, y_train), selector.transform(X_test)

def feature_search(X, y):
    return

    