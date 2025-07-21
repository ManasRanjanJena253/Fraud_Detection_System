import torch
from testing_dl_model import model
import pandas as pd
import pickle

data = pd.read_csv("data/test_data.csv")
print(data.sample(3))
device = "cuda" if torch.cuda.is_available() else "cpu"

X_train = data.drop(["Class"], axis = 1, inplace = False)
with open("saved_models/min_max_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
X_train_scaled = scaler.transform(X_train)
X_train_tensor = torch.tensor(X_train_scaled, dtype = torch.float32).to(device)

encoder_model = model
encoder_model.eval()

with torch.inference_mode():
    reconstructed = encoder_model(X_train_tensor)

# Calculating reconstructed error (RE)
# Using Mean Squared Error per sample (row-wise)
reconstruction_error = torch.mean((X_train_tensor - reconstructed) ** 2, dim=1)

# Creating a new column in original dataset for training the hybrid model
data["RE"] = reconstruction_error.cpu().numpy()   # Converting the device to cpu as tensors on gpu can't be converted into numpy arrays.

print(data.sample(3))

data.to_csv("data/hybrid_test_dataset.csv", index = False)
