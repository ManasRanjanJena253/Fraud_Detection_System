import pickle
import torch.nn as nn
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # Encoder: gradually compress
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4)  # Bottleneck
        )

        # Decoder: gradually reconstruct
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()  # Keep this for normalized input in [0, 1]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Load and prepare test data
test_data = pd.read_csv("data/test_data.csv")
X_test = test_data.drop(test_data[test_data["Class"] == 1].index, inplace=False)
X_test = X_test.drop(columns=["Class"])  # Droping the label column

# Applying MinMaxScaler
with open("saved_models/min_max_scaler.pkl", "rb") as f:
    min_max_scaler = pickle.load(f)

X_test_scaled = min_max_scaler.transform(X_test)

# Loading the model
input_dim = X_test_scaled.shape[1]
model = AutoEncoder(input_dim = input_dim)
model.load_state_dict(torch.load("saved_models/autoencoder_model.pth"))
model.eval()

# Move model and data to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Convert input to tensor
X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

# Forward pass (no gradient needed)
with torch.inference_mode():
    reconstructed = model(X_tensor).cpu().numpy()

# Compare one sample visually
idx = np.random.randint(0, len(X_test_scaled))
plt.figure(figsize=(12, 4))
plt.plot(X_test_scaled[idx], label='Original')
plt.plot(reconstructed[idx], label='Reconstructed')
plt.legend()
plt.title(f"Sample #{idx} Reconstruction")
plt.show()
