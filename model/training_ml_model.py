# Importing dependencies
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

data_res = pd.read_csv("data/resampled_train_data.csv")

X_train = data_res.drop(columns = ["Class"], axis = 1, inplace = False)
y_train = data_res["Class"]

test_data = pd.read_csv("data/test_data.csv")
X_test = test_data.drop(columns = ["Class"], axis = 1, inplace = False)
y_test = test_data["Class"]

print("X_train shape : ", X_train.shape)
print("y_train shape", y_train.shape)

base_models = [("xgb", XGBClassifier(n_jobs = -1, random_state = 21)),
               ("rf", RandomForestClassifier(n_jobs = -1, random_state = 21)),
               ("nb", GaussianNB())
               ]
meta_model = LogisticRegression(n_jobs = -1, random_state = 21)

model = StackingClassifier(estimators = base_models, final_estimator = meta_model, cv = 5, verbose = 2, n_jobs = -1)  # Verbose shows the training process.
print("Model Training Started")

model.fit(X_train, y_train)

with open("saved_models/standard_scaler.pkl", "rb") as f:
    std_scaler = pickle.load(f)

X_test_scaled = std_scaler.transform(X_test)

y_pred = model.predict(X_test_scaled)

print("========================================================================")
print()
print(classification_report(y_test, y_pred))
print()
print("=========================================================================")

# Save the model to a file
with open("saved_models/stacking_classifierV1.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model Saved")
