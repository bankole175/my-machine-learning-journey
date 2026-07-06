import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import r2_score

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.datasets import load_diabetes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data = load_diabetes()
X, y = data.data, data.target

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
scaler_X = StandardScaler()
X_train  = scaler_X.fit_transform(X_train)
X_val    = scaler_X.transform(X_val)

# Scale y — new for regression
scaler_y = StandardScaler()
y_train  = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_val    = scaler_y.transform(y_val.reshape(-1, 1)).flatten()

# Now convert to tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
y_val_t   = torch.tensor(y_val,   dtype=torch.float32)
print(f"Dataset shapes: train {X_train_t.shape}, val {X_val_t.shape}")
print(f"Label type: train {y_train_t.dtype}, val {y_train_t[:5]}")

train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset = TensorDataset(X_val_t, y_val_t)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
print(f"Batches per epoch: {len(train_loader)} train, {len(val_loader)} val")

class DiabetesRegression(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size // 2)
        self.layer3 = nn.Linear(hidden_size // 2, output_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)

    def forward(self, x):
        x = self.relu(self.bn1(self.layer1(x)))
        x = self.dropout(x)

        x = self.relu(self.bn2(self.layer2(x)))
        x = self.dropout(x)

        x = self.layer3(x)
        return x

model = DiabetesRegression(input_size=10, hidden_size=64, output_size=1)
model = model.to(device)

total = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total:,}")

loss_fn = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

best_val_r2 = 0.0
best_model_state = None
no_improve = 0
patience = 20

num_epochs = 300

for epoch in range(num_epochs):
    model.train()

    total_train_loss = 0.0
    train_preds_all = []
    train_labels_all = []

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        # FIX 1 & 2: Call model once, squeeze output to match 1D y_batch
        y_pred = model(X_batch).squeeze()
        loss = loss_fn(y_pred, y_batch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_train_loss += loss.item()

        # Reuse y_pred instead of calling model() again
        train_preds_all.extend(y_pred.cpu().detach().numpy().flatten())
        train_labels_all.extend(y_batch.cpu().numpy())

    avg_train_loss = total_train_loss / len(train_loader)
    r2 = r2_score(train_labels_all, train_preds_all)
    rmse = np.sqrt(np.mean((np.array(train_preds_all) - np.array(train_labels_all)) ** 2))

    # Optional print statement cleanup
    # print(f"Epoch {epoch+1} Train R²: {r2:.3f} | RMSE: {rmse:.1f}")

    model.eval()

    total_val_loss = 0.0
    val_preds_all = []
    val_labels_all = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # FIX: Call model once, flatten to safely protect against 0D array drops
            y_pred = model(X_batch).squeeze()

            # Catch edge case if batch size is 1 and squeeze() removes all dimensions
            if y_pred.dim() == 0:
                y_pred = y_pred.unsqueeze(0)

            loss = loss_fn(y_pred, y_batch)
            total_val_loss += loss.item()

            val_preds_all.extend(y_pred.cpu().numpy().flatten())
            val_labels_all.extend(y_batch.cpu().numpy())

    avg_val_loss = total_val_loss / len(val_loader)
    val_r2 = r2_score(val_labels_all, val_preds_all)
    val_rmse = np.sqrt(np.mean((np.array(val_preds_all) - np.array(val_labels_all)) ** 2))

    if val_r2 > best_val_r2:
        best_val_r2 = val_r2
        best_model_state = model.state_dict()
        no_improve = 0
        torch.save(model.state_dict(), "diabetes_model.pth")
        status = "improved"
    else:
        no_improve += 1
        status = f"no imp {no_improve}/{patience}"

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch + 1:3d} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val R²: {val_r2:.3f} | "
              f"{status}")

    if no_improve >= patience:
        print(f"Early stopping at epoch {epoch + 1}")
        break
if best_model_state is not None:
    model.load_state_dict(best_model_state)

torch.save(model.state_dict(), "diabetes_model.pth")
print("Model saved!")

model.eval()

final_preds = []
final_labels = []

with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        preds = logits.squeeze().cpu().numpy()
        final_preds.extend(preds)
        final_labels.extend(y_batch.cpu().numpy())

# Convert back to original scale before evaluating
final_preds_real  = scaler_y.inverse_transform(
    np.array(final_preds).reshape(-1, 1)
).flatten()
final_labels_real = scaler_y.inverse_transform(
    np.array(final_labels).reshape(-1, 1)
).flatten()

final_r2 = r2_score(final_labels_real, final_preds_real)
final_rmse = np.sqrt(np.mean((final_preds_real - final_labels_real) ** 2))

print(f"Final R²:   {final_r2:.3f}")
print(f"Final RMSE: {final_rmse:.1f}")

# Show a few actual vs predicted values
print("\nSample predictions:")
print(f"{'Actual':>10} {'Predicted':>10} {'Error':>10}")
print("-" * 32)
for actual, pred in zip(final_labels_real[:10], final_preds_real[:10]):
    error = abs(actual - pred)
    print(f"{actual:>10.1f} {pred:>10.1f} {error:>10.1f}")




