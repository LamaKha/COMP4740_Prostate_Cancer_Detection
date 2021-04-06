from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def classification_model(X_train, y_train, X_test, classifier):
    # Designate the classifier
    if (classifier == 'RF'):
        model = RandomForestClassifier(random_state = 42)
    elif (classifier == 'KNN'):
        model = KNN()
    elif (classifier == 'GNB'):
        model = GNB()
    elif (classifier == 'DT'):
        model = DecisionTreeClassifier()
    else:
        model = SVC()
        
    # Fit
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    return model, y_pred

def validate_model(model, y_pred, y_test):
    cm = confusion_matrix(y_test, y_pred)

    n = len(y_test)
    
    print(cm)
    correct = 0
    for i in range(len(cm)):
        correct = correct + cm[i][i]
    print (correct / n)
    
def classify_and_validate(X_train, y_train, X_test, y_test, classifier):
    model, y_pred = classification_model(X_train, y_train, X_test, classifier)
    validate_model(model, y_pred, y_test)