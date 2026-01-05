from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

class DiabetesModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000,
                                        class_weight="balanced")

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train.values.ravel())

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        print(classification_report(y_test, y_pred))
        print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba)}")

    def feature_importance(self, feature_names, top_k=10):
        coefficients = self.model.coef_[0]
        importance = list(zip(feature_names, coefficients))
        importance.sort(key=lambda x: abs(x[1]), reverse=True)
        return importance[:top_k]

    def predict(self, X_new):
        return self.model.predict(X_new)