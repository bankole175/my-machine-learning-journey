import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset

import numpy as np

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

X, y = make_blobs(n_samples=1000, centers=4, n_features=2, random_state=42)

print(X.shape)
print(y.shape)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Training labels: {np.unique(y_train)}")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.long)

print(f"Dataset shapes: train {X_train_t.shape}, val {X_val_t.shape}")
print(f"Label shapes: train {y_train_t.shape}, val {y_val_t.shape}")

train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset = TensorDataset(X_val_t, y_val_t)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"Batches per epoch: {len(train_loader)} train, {len(val_loader)} val")

class BlobClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size // 2)
        self.layer3 = nn.Linear(hidden_size // 2, output_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)

    def forward(self, x):
        x = self.relu(self.bn1(self.layer1(x)))
        x = self.dropout(x)

        x = self.relu(self.bn2(self.layer2(x)))
        x = self.dropout(x)

        x = self.layer3(x)
        return x

model = BlobClassifier(input_size=2, hidden_size=64, output_size=4)
model = model.to(device)

total = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total:,}")

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

best_val_f1 = 0.0
best_model_state = None
no_improve = 0
patience = 20

num_epochs = 100

for epoch in range(num_epochs):
    model.train()

    total_train_loss = 0.0
    train_preds_all = []
    train_labels_all = []

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits = model(X_batch)
        loss = loss_fn(logits, y_batch)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_train_loss += loss.item()

        preds = logits.argmax(dim=1).cpu().numpy()
        train_preds_all.extend(preds)
        train_labels_all.extend(y_batch.cpu().numpy())

    avg_train_loss = total_train_loss / len(train_loader)
    train_f1 = f1_score(train_labels_all, train_preds_all, average="weighted")

    model.eval()

    total_val_loss = 0.0
    val_preds_all = []
    val_labels_all = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)

            total_val_loss += loss.item()

            preds = logits.argmax(dim=1).cpu().numpy()
            val_preds_all.extend(preds)
            val_labels_all.extend(y_batch.cpu().numpy())

    avg_val_loss = total_val_loss / len(val_loader)
    val_f1 = f1_score(val_labels_all, val_preds_all, average="weighted")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_model_state = model.state_dict()
        no_improve = 0
        status = "improved"
    else:
        no_improve += 1
        status = f"no imp {no_improve}/{patience}"

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch + 1:3d} | "
              f"Train Loss: {avg_train_loss:.4f} | Train F1: {train_f1:.3f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.3f} | "
              f"{status}")
    if no_improve >= patience:
        print(f"Early stopping at epoch {epoch + 1}")
        break

if best_model_state is not None:
    model.load_state_dict(best_model_state)

torch.save(model.state_dict(), "blob_model.pth")
print("Model saved!")

model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        preds = logits.argmax(dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(y_batch.cpu().numpy())

report = classification_report(all_labels, all_preds, target_names=["Class 0", "Class 1", "Class 2", "Class 3"])
print(report)

model.eval()

with torch.no_grad():
    sample = X_val_t[0].unsqueeze(0).to(device)
    logits = model(sample)
    probs = logits.softmax(dim=1).cpu().numpy()

    predicted = logits.argmax(dim=1).item()
    confidence = probs[0][predicted].item()

print(f"Predicted class: {predicted}, Confidence: {confidence:.3f}")

checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "best_val_f1": best_val_f1,
}

torch.save(checkpoint, "blob_checkpoint.pth")
print("Checkpoint saved!")

loaded_model = BlobClassifier(input_size=2, hidden_size=64, output_size=4)
loaded_model = loaded_model.to(device)

loaded_checkpoint = torch.load("blob_checkpoint.pth", map_location=device)
loaded_model.load_state_dict(loaded_checkpoint["model_state_dict"])
optimizer.load_state_dict(loaded_checkpoint["optimizer_state_dict"])

start_epoch = loaded_checkpoint["epoch"] + 1
loaded_model.eval()

print(f"Checkpoint loaded. Resume from epoch {start_epoch}.")