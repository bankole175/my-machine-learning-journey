# ═══════════════════════════════════════════════════════════════
# BREAST CANCER CLASSIFIER — Complete Code
# Goal: Predict whether a tumour is malignant (0) or benign (1)
# Dataset: 569 patients, 30 measurements each
# ═══════════════════════════════════════════════════════════════


# ── STEP 1: IMPORTS ─────────────────────────────────────────────
# PyTorch — the ML framework we use for everything
import torch
import torch.nn as nn
import torch.optim as optim

# DataLoader handles batching and shuffling automatically
from torch.utils.data import DataLoader, TensorDataset

# sklearn — for loading data, splitting, scaling, and evaluating
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score


# ── STEP 2: DEVICE SETUP ────────────────────────────────────────
# Use GPU if available, otherwise CPU
# All data and model must be on the same device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ── STEP 3: LOAD DATA ───────────────────────────────────────────
# Load the breast cancer dataset — built into sklearn, no downloading needed
data = load_breast_cancer()

# X = features (569 patients × 30 measurements each)
# y = labels   (0 = malignant, 1 = benign)
X, y = data.data, data.target

print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Classes: {data.target_names}")


# ── STEP 4: SPLIT INTO TRAIN AND VALIDATION ─────────────────────
# test_size=0.2 means 80% train, 20% validation
# random_state=42 means the same split every time you run
# stratify=y means keep the same proportion of malignant/benign in each split
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size    = 0.2,
    random_state = 42,
    stratify     = y
)

print(f"Training samples:   {len(X_train)}")
print(f"Validation samples: {len(X_val)}")


# ── STEP 5: SCALE FEATURES ──────────────────────────────────────
# The 30 features are on very different scales (area: 500-2000, smoothness: 0.05-0.15)
# StandardScaler makes every feature have mean=0 and std=1
# This makes gradient descent work much faster and more reliably

scaler = StandardScaler()

# fit_transform on training data: learns the mean/std FROM training data
# and scales it. We learn the statistics from training only.
X_train = scaler.fit_transform(X_train)

# transform on validation: uses the SAME mean/std learned from training
# We never fit on validation — that would be cheating
X_val = scaler.transform(X_val)


# ── STEP 6: CONVERT TO PYTORCH TENSORS ─────────────────────────
# PyTorch only works with tensors — not NumPy arrays
# float32 for features — matches the model's weight type
# long (int64) for labels — CrossEntropyLoss requires integer class indices
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
y_val_t   = torch.tensor(y_val,   dtype=torch.long)

print(f"\nTensor shapes:")
print(f"  X_train: {X_train_t.shape}")   # (455, 30)
print(f"  y_train: {y_train_t.shape}")   # (455,)
print(f"  X_val:   {X_val_t.shape}")     # (114, 30)
print(f"  y_val:   {y_val_t.shape}")     # (114,)


# ── STEP 7: CREATE DATALOADERS ──────────────────────────────────
# TensorDataset pairs each X with its corresponding y
# DataLoader handles splitting into batches automatically

# batch_size=32: model sees 32 samples at a time before updating weights
# shuffle=True for training: different order each epoch — prevents learning the order
# shuffle=False for validation: order does not matter, just measuring performance
train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset   = TensorDataset(X_val_t,   y_val_t)

train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader    = DataLoader(val_dataset,   batch_size=32, shuffle=False)

print(f"\nBatches per epoch: {len(train_loader)} train, {len(val_loader)} val")


# ── STEP 8: DEFINE THE MODEL ────────────────────────────────────
# We inherit from nn.Module — the standard PyTorch way to define any model
class CancerClassifier(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        # Always call super().__init__() first — initialises nn.Module
        super().__init__()

        # Three linear layers — the core of the network
        # input_size=30  (30 tumour measurements)
        # hidden_size=64 (arbitrary — enough capacity for this problem)
        # hidden_size//2 = 32
        # output_size=2  (two classes: malignant or benign)
        self.layer1 = nn.Linear(input_size,      hidden_size)
        self.layer2 = nn.Linear(hidden_size,      hidden_size // 2)
        self.layer3 = nn.Linear(hidden_size // 2, output_size)

        # ReLU activation — without this, stacking linear layers is pointless
        # (three linear layers without activation = one linear layer)
        self.relu = nn.ReLU()

        # Dropout — randomly zeros 30% of neurons during training
        # Forces the model to learn redundant patterns — prevents memorisation
        self.dropout = nn.Dropout(0.3)

        # BatchNorm — normalises activations between layers
        # Makes training faster and more stable
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)

    def forward(self, x):
        # Layer 1: Linear → BatchNorm → ReLU → Dropout
        x = self.relu(self.bn1(self.layer1(x)))
        x = self.dropout(x)

        # Layer 2: Linear → BatchNorm → ReLU → Dropout
        x = self.relu(self.bn2(self.layer2(x)))
        x = self.dropout(x)

        # Layer 3: Linear only — no activation, no dropout
        # Output is raw scores (logits) for each class
        # CrossEntropyLoss applies Softmax internally
        x = self.layer3(x)
        return x


# ── STEP 9: INSTANTIATE MODEL, LOSS, OPTIMISER ──────────────────
# Creating the actual model object from the class blueprint
model = CancerClassifier(
    input_size  = 30,   # 30 tumour measurements
    hidden_size = 64,   # neurons in first hidden layer
    output_size = 2     # 2 classes to predict
)

# Move model to GPU if available, otherwise CPU
model = model.to(device)

# CrossEntropyLoss — correct loss function for classification
# Applies Softmax internally — do NOT apply Softmax yourself before this
# Expects raw logits as input and integer class labels as targets
loss_fn = nn.CrossEntropyLoss()

# AdamW — adaptive learning rate optimiser with L2 regularisation
# lr=0.001       — how big each weight update step is
# weight_decay   — L2 regularisation: penalises large weights to prevent overfitting
optimizer = optim.AdamW(
    model.parameters(),
    lr           = 0.001,
    weight_decay = 0.01
)

# Count total parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel created with {total_params:,} parameters")


# ── STEP 10: TRAINING LOOP ──────────────────────────────────────
print("\nStarting training...")
print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Train F1':>10} | "
      f"{'Val Loss':>10} | {'Val F1':>8} | {'Status':>8}")
print("─" * 68)

# Track best model
best_val_f1   = 0.0
best_epoch    = 0
no_improve    = 0
patience      = 15    # stop if no improvement for 15 epochs

epochs = 100

for epoch in range(epochs):

    # ────────────────────────────────────────────────────────────
    # TRAINING PHASE
    # ────────────────────────────────────────────────────────────
    # Set model to training mode
    # This activates dropout and puts BatchNorm in training mode
    model.train()

    # Track total loss and predictions across all batches
    total_train_loss = 0
    train_preds_all  = []
    train_labels_all = []

    # Loop over each batch — train_loader gives us (X_batch, y_batch)
    for X_batch, y_batch in train_loader:

        # Move batch to same device as model (GPU or CPU)
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Step 1: Clear gradients from previous batch
        # ALWAYS do this first — gradients accumulate otherwise
        optimizer.zero_grad()

        # Step 2: Forward pass — run data through the model
        # Gets raw scores (logits) for each class
        logits = model(X_batch)

        # Step 3: Compute loss — how wrong were the predictions?
        # CrossEntropyLoss applies Softmax to logits internally
        loss = loss_fn(logits, y_batch)

        # Step 4: Backward pass — compute gradients
        # Each weight gets a gradient telling it which way to move
        loss.backward()

        # Step 5: Gradient clipping — prevents exploding gradients
        # If any gradient is larger than 1.0, scale it down
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Step 6: Update weights — move each weight in direction of gradient
        optimizer.step()

        # Accumulate loss for this batch
        total_train_loss += loss.item()

        # Collect predictions for F1 calculation
        # argmax(dim=1) picks the class with the highest score
        preds = logits.argmax(dim=1).cpu().numpy()
        train_preds_all.extend(preds)
        train_labels_all.extend(y_batch.cpu().numpy())

    # Average loss over all batches
    avg_train_loss = total_train_loss / len(train_loader)

    # F1 score on training data
    train_f1 = f1_score(train_labels_all, train_preds_all, average='weighted')

    # ────────────────────────────────────────────────────────────
    # VALIDATION PHASE
    # ────────────────────────────────────────────────────────────
    # Set model to evaluation mode
    # This DEACTIVATES dropout (all neurons active)
    # and puts BatchNorm in eval mode (uses running statistics)
    model.eval()

    total_val_loss  = 0
    val_preds_all   = []
    val_labels_all  = []

    # torch.no_grad() — do not track operations for backprop
    # Saves memory and speeds up evaluation
    # We never do backprop during validation
    with torch.no_grad():

        for X_batch, y_batch in val_loader:

            # Move batch to device
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward pass only — no backward pass needed
            logits = model(X_batch)

            # Compute loss for monitoring
            loss = loss_fn(logits, y_batch)
            total_val_loss += loss.item()

            # Convert logits to class predictions
            # argmax picks the index of the highest score
            preds = logits.argmax(dim=1).cpu().numpy()
            val_preds_all.extend(preds)
            val_labels_all.extend(y_batch.cpu().numpy())

    avg_val_loss = total_val_loss / len(val_loader)
    val_f1       = f1_score(val_labels_all, val_preds_all, average='weighted')

    # ────────────────────────────────────────────────────────────
    # TRACK BEST MODEL AND EARLY STOPPING
    # ────────────────────────────────────────────────────────────
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_epoch  = epoch + 1
        no_improve  = 0
        # Save the best model weights
        torch.save(model.state_dict(), 'best_cancer_model.pth')
        status = "✅ saved"
    else:
        no_improve += 1
        status = f"no imp {no_improve}"

    # Print every 5 epochs and the first epoch
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"{epoch+1:>6} | {avg_train_loss:>12.4f} | {train_f1:>10.3f} | "
              f"{avg_val_loss:>10.4f} | {val_f1:>8.3f} | {status:>8}")

    # Stop training if no improvement for patience epochs
    if no_improve >= patience:
        print(f"\nEarly stopping triggered at epoch {epoch+1}")
        break


# ── STEP 11: FINAL EVALUATION ───────────────────────────────────
print(f"\nBest model: Epoch {best_epoch} | Val F1: {best_val_f1:.3f}")

# Load the best model weights
model.load_state_dict(torch.load('best_cancer_model.pth'))

# Run final evaluation on validation set
model.eval()
final_preds  = []
final_labels = []

with torch.no_grad():
    for X_batch, y_batch in val_loader:
        logits = model(X_batch.to(device))
        preds  = logits.argmax(dim=1).cpu().numpy()
        final_preds.extend(preds)
        final_labels.extend(y_batch.numpy())

# Full report — precision, recall, F1 per class
print("\nFinal Classification Report:")
print(classification_report(
    final_labels,
    final_preds,
    target_names = ['Malignant', 'Benign']
))


# ── STEP 12: TEST ON A SINGLE SAMPLE ────────────────────────────
# This is what using the model in production looks like
print("Predicting on one real sample:")

# Take one sample from validation set
one_sample = X_val_t[0].unsqueeze(0).to(device)   # shape (1, 30)

model.eval()
with torch.no_grad():
    logits       = model(one_sample)               # shape (1, 2)
    probs        = torch.softmax(logits, dim=1)    # convert to probabilities
    predicted    = logits.argmax(dim=1).item()     # get class index
    confidence   = probs[0][predicted].item()      # get confidence

class_names  = ['Malignant', 'Negative']
actual_label = y_val[0]

print(f"  Predicted: {class_names[predicted]} ({confidence:.1%} confident)")
print(f"  Actual:    {class_names[actual_label]}")
print(f"  Correct:   {'✅' if predicted == actual_label else '❌'}")