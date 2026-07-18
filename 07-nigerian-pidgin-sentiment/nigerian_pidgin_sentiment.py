import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, f1_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
df = pd.read_csv('combined_train.csv')
texts  = df['text'].tolist()
labels = df['label'].tolist()

label_map = {'positive': 0, 'negative': 1, 'neutral': 2}
y = np.array([label_map[label] for label in labels])

# Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(texts, y, test_size=0.2, random_state=42, stratify=y)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
X_train    = vectorizer.fit_transform(X_train).toarray()
X_val      = vectorizer.transform(X_test).toarray()

# Convert EVERYTHING to tensors safely
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
x_val_t   = torch.tensor(X_val,   dtype=torch.float32)
y_val_t   = torch.tensor(y_test,  dtype=torch.long)

# Setup clean DataLoaders
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)

val_dataset = TensorDataset(x_val_t, y_val_t)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Model definition remains unchanged
class SentimentClassifier(nn.Module):
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

model = SentimentClassifier(input_size=X_train.shape[1], hidden_size=32, output_size=3).to(device)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array([0, 1, 2]),   # positive, negative, neutral per your label_map
    y=y_train
)
class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)
print(f"Class weights (positive, negative, neutral): {class_weights}")

loss_fn = nn.CrossEntropyLoss(weight=class_weights_t)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

best_val_f1 = 0.0
best_model_state = None
no_improve = 0
patience = 20
num_epochs = 800
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0.0
    train_preds_all = []
    train_labels_all = []

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = loss_fn(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        train_preds_all.extend(preds)
        train_labels_all.extend(y_batch.cpu().numpy())

    avg_train_loss = total_train_loss / len(train_loader)
    train_f1 = f1_score(train_labels_all, train_preds_all, average="weighted")

    # Evaluation
    model.eval()
    total_val_loss = 0.0
    val_preds_all = []
    val_labels_all = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)

            total_val_loss += loss.item()
            preds = logits.argmax(dim=1).cpu().numpy()
            val_preds_all.extend(preds)
            val_labels_all.extend(y_batch.cpu().numpy())

    avg_val_loss = total_val_loss / max(1, len(val_loader))
    val_f1 = f1_score(val_labels_all, val_preds_all, average="weighted")

    if val_f1 > best_val_f1 and avg_val_loss <= best_val_loss * 1.15:
        best_val_f1 = val_f1
        best_val_loss = avg_val_loss
        best_model_state = model.state_dict()
        no_improve = 0
        status = "improved"
    else:
        no_improve += 1
        status = f"no imp {no_improve}/{patience}"

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch + 1:3d} | Train Loss: {avg_train_loss:.4f} | Train F1: {train_f1:.3f} | Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.3f} | {status}")

    if no_improve >= patience:
        print(f"Early stopping at epoch {epoch + 1}")
        break

if best_model_state is not None:
    model.load_state_dict(best_model_state)

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.cpu().numpy())

report = classification_report(all_labels, all_preds, target_names=['positive', 'negative', 'neutral'], zero_division=0)
print(report)

real_df = pd.read_csv('naijasenti_test.csv')
print(f"Loaded real test set: {len(real_df)} rows, label counts:\n{real_df['label'].value_counts()}")

real_texts = real_df['text'].tolist()
real_labels = real_df['label'].tolist()
y_real = np.array([label_map[label] for label in real_labels])

# IMPORTANT: transform only, do NOT fit_transform — must reuse training vocabulary
X_real = vectorizer.transform(real_texts).toarray()
X_real_t = torch.tensor(X_real, dtype=torch.float32)
y_real_t = torch.tensor(y_real, dtype=torch.long)

real_dataset = TensorDataset(X_real_t, y_real_t)
real_loader = DataLoader(real_dataset, batch_size=64, shuffle=False)

model.eval()
real_preds_all = []
real_labels_all = []

with torch.no_grad():
    for X_batch, y_batch in real_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        preds = logits.argmax(dim=1).cpu().numpy()
        real_preds_all.extend(preds)
        real_labels_all.extend(y_batch.cpu().numpy())

print("\n=== Evaluation on REAL, unseen Pidgin tweets (naijasenti_test.csv) ===")
real_report = classification_report(
    real_labels_all, real_preds_all,
    target_names=['positive', 'negative', 'neutral'],
    zero_division=0
)
print(real_report)

# Sanity check: how much of the real vocabulary did TF-IDF actually recognize?
real_vecs = vectorizer.transform(real_texts)
nonzero_rows = (real_vecs.sum(axis=1) > 0).sum()
print(f"Real test rows with at least one recognized TF-IDF feature: {nonzero_rows}/{len(real_texts)}")

# Checkpoint Structure
checkpoint = {
    'epoch': epoch,
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'best_val_f1': best_val_f1,
}
torch.save(checkpoint, "nigerian_pidgin_sentiment.pth")
print("Checkpoint saved successfully!")
