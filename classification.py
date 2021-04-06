from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


def classification_model(X_train, y_train, X_test, classifier):
    if (classifier == 'RF'):
        model = RandomForestClassifier(random_state = 42)

    # Fit
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    return model, y_pred

def validate_model(model, y_pred, y_test):
    cm = confusion_matrix(y_test, y_pred)

    n = len(y_test)
    
    print(cm)
    print(n)
    
def classify_and_validate(X_train, y_train, X_test, y_test, classifier):
    model, y_pred = classification_model(X_train, y_train, X_test, classifier)
    validate_model(model, y_pred, y_test)