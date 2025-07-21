# Importing dependencies
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier

data_res = pd.read_csv("data/hybrid_test_dataset.csv")

X = data_res.drop(columns = ["Class"], axis = 1, inplace = False)
y = data_res["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.2, random_state = 21)

print("X_train shape : ", X_train.shape)
print("y_train shape", y_train.shape)

base_models = [("xgb", XGBClassifier(n_jobs = -1, random_state = 21)),
               ("rf", RandomForestClassifier(n_jobs = -1, random_state = 21)),
               ("logistic_regression", LogisticRegression(n_jobs = -1, random_state = 21, max_iter = 5000))
               ]
meta_model = GaussianNB()

model = StackingClassifier(estimators = base_models, final_estimator = meta_model, cv = 5, verbose = 8, n_jobs = -1)  # Verbose shows the training process.
print("Model Training Started")

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("========================================================================")
print()
print(classification_report(y_test, y_pred))
print()
print("=========================================================================")

# Save the model to a file
with open("saved_models/hybrid_classifierV1.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model Saved")
