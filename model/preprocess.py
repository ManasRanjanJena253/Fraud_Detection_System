# Importing dependencies
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SVMSMOTE  # Using SVMSMOTE because it is much better for extreme class imbalance in binary classification. And it creates a better decision boundary which will help with classification.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("data/creditcard.csv")

# Getting insights about the data
print(data.head(10))
print(data.isnull().sum())
print(data.info())

# Oversampling the data to handle class imbalance
X = data.drop(columns = ["Class"], axis = 1, inplace = False)
y = data["Class"]
print("X shape :", X.shape)
print("y shape :", y.shape)

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 21)

test_data = np.concatenate((X_test, y_test.to_numpy().reshape(-1, 1)), axis = -1, )
test_df = pd.DataFrame(test_data, columns = data.columns.tolist())
test_df.to_csv("data/test_data.csv", index = False)

std_scaler = StandardScaler()
X_scaled = std_scaler.fit_transform(X_train)
with open("saved_models/standard_scaler.pkl", "wb") as f:
    pickle.dump(std_scaler, f)

sm = SVMSMOTE(random_state = 21)
X_res, y_res = sm.fit_resample(X_scaled, y_train)

all_data = np.concatenate((X_res, y_res.to_numpy().reshape(-1, 1)), axis = 1)
data_res = pd.DataFrame(all_data, columns = data.columns.tolist())
data_res.to_csv("data/resampled_train_data.csv", index = False)

# Checking the data distribution after resampling
sns.displot(data_res["Class"])
plt.show()
