import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm

torch.backends.cudnn.benchmark = True

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


# Preparing the data
data = pd.read_csv("data/resampled_train_data.csv")
valid_transactions_data = data.drop(data[data["Class"] == 1].index, inplace = False)   # Keeping only the valid transaction, so the autoencoder only creates valid transactions.
X = valid_transactions_data.drop(["Class"], axis = 1, inplace = False)

# Converting the arrays to tensors
X_train_tensor = torch.tensor(X.values, dtype = torch.float32)

# Hyperparameters
input_dim = X_train_tensor.shape[1]
batch_size = 256
num_epochs = 100
lr = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {device}")

# Loading the data
train_dataset = TensorDataset(X_train_tensor)
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = 0)

model = AutoEncoder(input_dim = input_dim)
model.to(device)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = lr)

# Training loop
model.train()
for epoch in tqdm(range(num_epochs)):
    epoch_loss = 0

    for batch in train_loader:
        x = batch[0].to(device)

        # Forward pass
        reconstruct = model(x)
        loss = loss_fn(reconstruct, x)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    if epoch % 20 == 0:
        print(f"Epoch : {epoch + 1}, Loss : {avg_loss}")
    elif epoch == num_epochs - 1:
        print(f"Final Loss : {avg_loss}")

# Saving the model
torch.save(model.state_dict(), "saved_models/autoencoder_model.pth")
