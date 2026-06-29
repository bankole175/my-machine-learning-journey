"""
=======================================================================
BACKPROPAGATION — COMPLETE STUDY CODE
Samuel Bankole — AI Research Learning Plan
=======================================================================

This file walks through backpropagation completely from scratch.
Run it section by section, read every comment, and study the output.

Sections:
  1. The building blocks — activations and their derivatives
  2. Manual backprop on a single neuron
  3. Full neural network from scratch — XOR problem
  4. Visualising the training process
  5. Common problems and fixes
  6. The same network in PyTorch — compare results
  7. Verifying manual gradients match PyTorch gradients exactly

Requirements: numpy, matplotlib, torch
Install:  pip install numpy matplotlib torch
=======================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

np.random.seed(42)
torch.manual_seed(42)

print("=" * 60)
print("BACKPROPAGATION — COMPLETE STUDY CODE")
print("Samuel Bankole — AI Research Learning Plan")
print("=" * 60)


# ═══════════════════════════════════════════════════════════════════
# SECTION 1 — THE BUILDING BLOCKS
# Activation functions and their derivatives
# ═══════════════════════════════════════════════════════════════════
print("\n\n" + "═" * 60)
print("SECTION 1 — ACTIVATION FUNCTIONS & DERIVATIVES")
print("═" * 60)

def sigmoid(z):
    """
    Squashes any number into (0, 1) — used for binary output.
    At z=0, sigmoid(0) = 0.5
    At z=+∞, sigmoid → 1
    At z=-∞, sigmoid → 0
    """
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    """
    The derivative of sigmoid is sigmoid(z) * (1 - sigmoid(z)).
    Maximum value is 0.25 at z=0.
    This small max value causes vanishing gradients in deep nets.
    """
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    """
    Rectified Linear Unit — most common hidden layer activation.
    Passes positive values unchanged, kills negative values.
    Simple but very effective.
    """
    return np.maximum(0, z)

def relu_derivative(z):
    """
    Derivative of ReLU:
    - 1 where z > 0 (gradient passes through unchanged)
    - 0 where z <= 0 (gradient is killed — "dying ReLU" problem)
    """
    return (z > 0).astype(float)

def leaky_relu(z, alpha=0.01):
    """
    Fix for dying ReLU — small slope for negative values.
    """
    return np.where(z > 0, z, alpha * z)

def leaky_relu_derivative(z, alpha=0.01):
    return np.where(z > 0, 1.0, alpha)

def binary_cross_entropy(predictions, targets):
    """
    Loss function for binary classification.
    Penalises confident wrong predictions very heavily.
    Minimising this = maximising likelihood of correct labels.
    """
    return -np.mean(
        targets * np.log(predictions + 1e-8) +
        (1 - targets) * np.log(1 - predictions + 1e-8)
    )

# --- Show activation values and derivatives ---
z_values = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])

print("\nSigmoid and its derivative:")
print(f"{'z':>6} | {'sigmoid(z)':>12} | {'sigmoid_deriv(z)':>18}")
print("-" * 44)
for z in z_values:
    print(f"{z:>6.1f} | {sigmoid(z):>12.6f} | {sigmoid_derivative(z):>18.6f}")

print("\nReLU and its derivative:")
print(f"{'z':>6} | {'relu(z)':>10} | {'relu_deriv(z)':>15}")
print("-" * 36)
for z in z_values:
    print(f"{z:>6.1f} | {relu(z):>10.4f} | {relu_derivative(z):>15.4f}")

print("\n✅ Key insight: ReLU derivative is 0 or 1.")
print("   Sigmoid derivative max is 0.25 — causes vanishing gradients in deep nets.")
print("   That's why we use ReLU in hidden layers, not sigmoid.")


# ═══════════════════════════════════════════════════════════════════
# SECTION 2 — MANUAL BACKPROP ON A SINGLE NEURON
# The simplest possible case — understand this first
# ═══════════════════════════════════════════════════════════════════
print("\n\n" + "═" * 60)
print("SECTION 2 — BACKPROP ON A SINGLE NEURON")
print("═" * 60)

print("""
Single neuron setup:
  Input:      x = 2.0
  Weight:     w = 0.5
  Bias:       b = -1.0
  True label: y = 1.0

  Forward:  z = w*x + b = 0.5*2 + (-1) = 0.0
            pred = sigmoid(z) = sigmoid(0.0) = 0.5
            loss = BCE(0.5, 1.0)

  Backward: How should w and b change to make pred closer to 1.0?
""")

# Forward pass
x = 2.0
w = 0.5
b = -1.0
y = 1.0

z    = w * x + b           # Linear step
pred = sigmoid(z)           # Activation
loss = binary_cross_entropy(np.array([pred]), np.array([y]))

print(f"Forward pass:")
print(f"  z    = w*x + b = {w}*{x} + {b} = {z}")
print(f"  pred = sigmoid({z}) = {pred:.6f}")
print(f"  loss = BCE({pred:.4f}, {y}) = {loss:.6f}")

# Backward pass — chain rule
# dLoss/dpred = -(y/pred - (1-y)/(1-pred))   [derivative of BCE]
# dpred/dz    = sigmoid_derivative(z)          [derivative of sigmoid]
# dz/dw       = x                              [derivative of z w.r.t. w]
# dz/db       = 1                              [derivative of z w.r.t. b]

dLoss_dpred = -(y / (pred + 1e-8) - (1 - y) / (1 - pred + 1e-8))
dpred_dz    = sigmoid_derivative(z)
dLoss_dz    = dLoss_dpred * dpred_dz    # Chain rule: multiply derivatives

dLoss_dw = dLoss_dz * x    # Chain rule: dz/dw = x
dLoss_db = dLoss_dz * 1    # Chain rule: dz/db = 1

print(f"\nBackward pass (chain rule):")
print(f"  dLoss/dpred = {dLoss_dpred:.6f}")
print(f"  dpred/dz    = {dpred_dz:.6f}  (sigmoid derivative)")
print(f"  dLoss/dz    = {dLoss_dz:.6f}  (chain rule: multiply above two)")
print(f"  dLoss/dw    = {dLoss_dw:.6f}  (chain rule: multiply by x={x})")
print(f"  dLoss/db    = {dLoss_db:.6f}  (chain rule: multiply by 1)")

# Weight update
lr = 0.1
w_new = w - lr * dLoss_dw
b_new = b - lr * dLoss_db

pred_new = sigmoid(w_new * x + b_new)

print(f"\nWeight update (lr={lr}):")
print(f"  w: {w:.4f} → {w_new:.4f}  (changed by {w_new-w:.6f})")
print(f"  b: {b:.4f} → {b_new:.4f}  (changed by {b_new-b:.6f})")
print(f"  New prediction: {pred_new:.6f}  (was {pred:.6f}, moved toward 1.0 ✅)")


# ═══════════════════════════════════════════════════════════════════
# SECTION 3 — FULL NEURAL NETWORK FROM SCRATCH
# Solving XOR — impossible with a single layer, needs backprop
# ═══════════════════════════════════════════════════════════════════
print("\n\n" + "═" * 60)
print("SECTION 3 — FULL NEURAL NETWORK: XOR PROBLEM")
print("═" * 60)

print("""
XOR truth table — cannot be separated by a straight line:
  [0, 0] → 0
  [0, 1] → 1
  [1, 0] → 1
  [1, 1] → 0

Architecture:
  Input layer:  2 neurons
  Hidden layer: 4 neurons (ReLU)
  Output layer: 1 neuron  (Sigmoid)
""")

# Dataset
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]], dtype=float)

y = np.array([[0],
              [1],
              [1],
              [0]], dtype=float)

# Initialise weights
# Small random values — NOT zeros (zeros cause symmetry: all neurons learn same thing)
W1 = np.random.randn(2, 4) * 0.1   # (input_size, hidden_size)
b1 = np.zeros((1, 4))               # (1, hidden_size)
W2 = np.random.randn(4, 1) * 0.1   # (hidden_size, output_size)
b2 = np.zeros((1, 1))               # (1, output_size)

print(f"Initial weights:")
print(f"  W1 shape: {W1.shape}  (2 inputs → 4 hidden neurons)")
print(f"  b1 shape: {b1.shape}")
print(f"  W2 shape: {W2.shape}  (4 hidden → 1 output neuron)")
print(f"  b2 shape: {b2.shape}")


def forward_pass(X, W1, b1, W2, b2):
    """
    Run data through the network layer by layer.
    Cache everything — needed for backward pass.

    Layer 1: Z1 = X @ W1 + b1   (linear transformation)
             A1 = ReLU(Z1)       (non-linear activation)

    Layer 2: Z2 = A1 @ W2 + b2  (linear transformation)
             A2 = sigmoid(Z2)    (output probability)
    """
    # Hidden layer
    Z1 = X @ W1 + b1      # Shape: (4, 4) — 4 samples, 4 hidden neurons
    A1 = relu(Z1)          # Shape: (4, 4)

    # Output layer
    Z2 = A1 @ W2 + b2     # Shape: (4, 1) — 4 samples, 1 output
    A2 = sigmoid(Z2)       # Shape: (4, 1) — probabilities

    cache = (X, Z1, A1, Z2, A2)
    return A2, cache


def backward_pass(cache, y, W2):
    """
    Compute gradients using chain rule — moving BACKWARD through layers.

    Key formula at each step:
      gradient = (gradient from next layer) * (local derivative)

    This is the chain rule: dL/dW = dL/dA * dA/dZ * dZ/dW
    """
    X, Z1, A1, Z2, A2 = cache
    m = len(y)   # Number of samples — divide by m to get mean gradient

    # ── OUTPUT LAYER ─────────────────────────────────────────────
    # Gradient of loss w.r.t. A2
    # For BCE + sigmoid, this simplifies beautifully to (pred - true)
    dA2 = A2 - y                         # Shape: (4, 1)

    # Gradient w.r.t. Z2
    # dL/dZ2 = dL/dA2 * dA2/dZ2
    # dA2/dZ2 = sigmoid_derivative — but combined with BCE it simplifies to dA2
    dZ2 = dA2                            # Shape: (4, 1)

    # Gradient w.r.t. W2
    # dL/dW2 = A1.T @ dZ2  (each hidden neuron's contribution to output error)
    dW2 = A1.T @ dZ2 / m                # Shape: (4, 1)

    # Gradient w.r.t. b2
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m   # Shape: (1, 1)

    # ── HIDDEN LAYER ──────────────────────────────────────────────
    # Error signal flows backward through W2
    # dL/dA1 = dZ2 @ W2.T  (how much each hidden neuron contributed to output error)
    dA1 = dZ2 @ W2.T                    # Shape: (4, 4)

    # Chain rule through ReLU
    # dL/dZ1 = dL/dA1 * dA1/dZ1
    # dA1/dZ1 = relu_derivative — passes gradient where z > 0, kills it where z <= 0
    dZ1 = dA1 * relu_derivative(Z1)     # Shape: (4, 4)

    # Gradient w.r.t. W1
    # dL/dW1 = X.T @ dZ1  (each input's contribution to hidden layer error)
    dW1 = X.T @ dZ1 / m                # Shape: (2, 4)

    # Gradient w.r.t. b1
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m   # Shape: (1, 4)

    return dW1, db1, dW2, db2


def update_weights(W1, b1, W2, b2, dW1, db1, dW2, db2, lr):
    """
    Gradient descent update — move weights OPPOSITE to gradient.
    Opposite because we want to MINIMISE loss.
    """
    W1 = W1 - lr * dW1
    b1 = b1 - lr * db1
    W2 = W2 - lr * dW2
    b2 = b2 - lr * db2
    return W1, b1, W2, b2


def train_from_scratch(X, y, W1, b1, W2, b2, lr=0.5, epochs=10000):
    losses = []

    for epoch in range(epochs):
        # 1. Forward pass
        A2, cache = forward_pass(X, W1, b1, W2, b2)

        # 2. Compute loss
        loss = binary_cross_entropy(A2, y)
        losses.append(loss)

        # 3. Backward pass — compute gradients
        dW1, db1, dW2, db2 = backward_pass(cache, y, W2)

        # 4. Update weights
        W1, b1, W2, b2 = update_weights(
            W1, b1, W2, b2,
            dW1, db1, dW2, db2,
            lr
        )

        # Print progress
        if epoch % 1000 == 0:
            pred_labels = (A2 > 0.5).astype(int)
            accuracy = np.mean(pred_labels == y)
            print(f"  Epoch {epoch:5d} | Loss: {loss:.6f} | Accuracy: {accuracy:.2%}")

    return W1, b1, W2, b2, losses


print("\nTraining from scratch:\n")
W1, b1, W2, b2, losses = train_from_scratch(
    X, y, W1, b1, W2, b2, lr=0.5, epochs=10000
)

# Final evaluation
A2_final, _ = forward_pass(X, W1, b1, W2, b2)

print(f"\nFinal predictions after training:")
print(f"{'Input':>10} | {'True Label':>12} | {'Prediction':>12} | {'Correct':>8}")
print("-" * 52)
for i in range(len(X)):
    pred  = A2_final[i][0]
    true  = y[i][0]
    label = "✅" if (pred > 0.5) == bool(true) else "❌"
    print(f"{str(X[i].astype(int)):>10} | {int(true):>12} | {pred:>12.6f} | {label:>8}")


# ═══════════════════════════════════════════════════════════════════
# SECTION 4 — VISUALISING TRAINING
# Watch the loss curve — this is what you check in every project
# ═══════════════════════════════════════════════════════════════════
print("\n\n" + "═" * 60)
print("SECTION 4 — VISUALISING TRAINING")
print("═" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Backpropagation Training — XOR Problem", fontsize=14, fontweight='bold')

# Loss curve
axes[0].plot(losses, color='#2563EB', linewidth=2)
axes[0].set_title("Training Loss Over Time")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Binary Cross-Entropy Loss")
axes[0].set_yscale('log')
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=0.01, color='red', linestyle='--', alpha=0.5, label='Target: 0.01')
axes[0].legend()

# Decision boundary visualisation
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200),
                      np.linspace(-0.5, 1.5, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
A2_grid, _ = forward_pass(grid, W1, b1, W2, b2)
Z = A2_grid.reshape(xx.shape)

axes[1].contourf(xx, yy, Z, levels=50, cmap='RdBu_r', alpha=0.8)
axes[1].contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
axes[1].scatter(X[:, 0], X[:, 1], c=y.ravel(),
                cmap='RdBu_r', s=200, edgecolors='black',
                linewidth=2, zorder=5)
axes[1].set_title("Decision Boundary Learned by Network")
axes[1].set_xlabel("Input 1")
axes[1].set_ylabel("Input 2")

for i, (xi, yi) in enumerate(zip(X, y)):
    axes[1].annotate(f"({int(xi[0])},{int(xi[1])})={int(yi[0])}",
                     (xi[0], xi[1]),
                     textcoords="offset points",
                     xytext=(10, 10), fontsize=9)

plt.tight_layout()
plt.savefig('backprop_training.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Training plot saved as 'backprop_training.png'")
print("   Loss curve: should show steady decline toward 0")
print("   Decision boundary: network learned a non-linear boundary for XOR")


# ═══════════════════════════════════════════════════════════════════
# SECTION 5 — COMMON PROBLEMS AND FIXES
# What goes wrong in practice and how to fix it
# ═══════════════════════════════════════════════════════════════════
print("\n\n" + "═" * 60)
print("SECTION 5 — COMMON PROBLEMS AND FIXES")
print("═" * 60)

print("\n--- Problem 1: Vanishing Gradients ---")
print("Multiplying many sigmoid derivatives (max 0.25) shrinks gradients to near zero.")
print("Early layers stop learning entirely.\n")

# Demonstrate vanishing gradients
n_layers = 10
gradient = 1.0
print(f"Gradient shrinkage through {n_layers} sigmoid layers:")
for i in range(n_layers):
    gradient *= 0.25   # Worst case sigmoid derivative
    print(f"  After layer {i+1}: gradient = {gradient:.8f}")

print(f"\n  Gradient after {n_layers} layers: {gradient:.2e}")
print("  Solution: Use ReLU in hidden layers — derivative is 1, not 0.25")

print("\n--- Problem 2: Exploding Gradients ---")
print("Gradients grow exponentially — loss goes to infinity.\n")

gradient = 1.0
print(f"Gradient explosion through 10 layers (bad init):")
for i in range(10):
    gradient *= 2.0   # Gradient doubling each layer
    if gradient > 1e6:
        print(f"  After layer {i+1}: gradient = {gradient:.2e} 💥 EXPLODED")
        break
    print(f"  After layer {i+1}: gradient = {gradient:.4f}")

print("\n  Solution 1: Gradient clipping — cap gradients at max_norm")
print("  Solution 2: Proper weight initialisation (He init for ReLU)")

# He initialisation
def he_init(n_in, n_out):
    """He initialisation — designed for ReLU networks."""
    return np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)

W1_he = he_init(2, 4)
W2_he = he_init(4, 1)
print(f"\n  He init W1 std: {W1_he.std():.4f}  (should be ≈ √(2/2) = {np.sqrt(2/2):.4f})")
print(f"  He init W2 std: {W2_he.std():.4f}  (should be ≈ √(2/4) = {np.sqrt(2/4):.4f})")

print("\n--- Problem 3: Wrong Learning Rate ---")
print("Too high: loss explodes. Too low: convergence takes forever.\n")

for lr_test in [0.001, 0.1, 0.5, 2.0]:
    W1_t = np.random.randn(2, 4) * 0.1
    b1_t = np.zeros((1, 4))
    W2_t = np.random.randn(4, 1) * 0.1
    b2_t = np.zeros((1, 1))

    losses_test = []
    stable = True

    for epoch in range(500):
        A2_t, cache_t = forward_pass(X, W1_t, b1_t, W2_t, b2_t)
        loss_t = binary_cross_entropy(A2_t, y)

        if np.isnan(loss_t) or loss_t > 100:
            stable = False
            break

        losses_test.append(loss_t)
        dW1_t, db1_t, dW2_t, db2_t = backward_pass(cache_t, y, W2_t)
        W1_t, b1_t, W2_t, b2_t = update_weights(
            W1_t, b1_t, W2_t, b2_t,
            dW1_t, db1_t, dW2_t, db2_t, lr_test
        )

    if stable and losses_test:
        final_loss = losses_test[-1]
        status = "✅ Good" if final_loss < 0.1 else "⚠️ Slow"
        print(f"  lr={lr_test:.3f} | Final loss: {final_loss:.4f} | {status}")
    else:
        print(f"  lr={lr_test:.3f} | Loss exploded 💥 | Too high")

print("\n--- Problem 4: Zero Weight Initialisation ---")
print("If all weights are 0, all neurons compute the same gradient.")
print("They all learn the same thing — the network never uses its full capacity.\n")

W1_zero = np.zeros((2, 4))
b1_zero = np.zeros((1, 4))
W2_zero = np.zeros((4, 1))
b2_zero = np.zeros((1, 1))

A2_zero, cache_zero = forward_pass(X, W1_zero, b1_zero, W2_zero, b2_zero)
dW1_zero, _, _, _ = backward_pass(cache_zero, y, W2_zero)

print(f"  With zero init — all gradients in dW1 are identical:")
print(f"  dW1 =\n{dW1_zero}")
print("\n  Every column is the same — all neurons update identically.")
print("  Solution: always use random initialisation.")


# ═══════════════════════════════════════════════════════════════════
# SECTION 6 — THE SAME NETWORK IN PYTORCH
# See how PyTorch handles what you wrote manually
# ═══════════════════════════════════════════════════════════════════
print("\n\n" + "═" * 60)
print("SECTION 6 — SAME NETWORK IN PYTORCH")
print("═" * 60)

print("\nPyTorch does everything you wrote manually — automatically.")
print("loss.backward() is the entire backward pass in one line.\n")

# Convert data to PyTorch tensors
X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.float32)

# Define the same network architecture
torch.manual_seed(42)
model = nn.Sequential(
    nn.Linear(2, 4),    # W1, b1 — same as our manual W1, b1
    nn.ReLU(),
    nn.Linear(4, 1),    # W2, b2 — same as our manual W2, b2
    nn.Sigmoid()
)

optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
loss_fn   = nn.BCELoss()

pytorch_losses = []

print("Training with PyTorch:\n")
for epoch in range(10000):
    # Forward pass — automatic
    predictions = model(X_t)

    # Loss — automatic
    loss = loss_fn(predictions, y_t)
    pytorch_losses.append(loss.item())

    # Backward pass — ONE LINE does everything your manual code did
    optimizer.zero_grad()   # Clear gradients from previous step
    loss.backward()         # THIS IS BACKPROPAGATION — PyTorch computes all gradients

    # Update weights — automatic
    optimizer.step()

    if epoch % 1000 == 0:
        pred_labels = (predictions > 0.5).float()
        accuracy    = (pred_labels == y_t).float().mean()
        print(f"  Epoch {epoch:5d} | Loss: {loss.item():.6f} | Accuracy: {accuracy:.2%}")

print(f"\nFinal PyTorch predictions:")
with torch.no_grad():
    final_pred = model(X_t)

print(f"{'Input':>10} | {'True Label':>12} | {'Prediction':>12} | {'Correct':>8}")
print("-" * 52)
for i in range(len(X)):
    pred  = final_pred[i][0].item()
    true  = y[i][0]
    label = "✅" if (pred > 0.5) == bool(true) else "❌"
    print(f"{str(X[i].astype(int)):>10} | {int(true):>12} | {pred:>12.6f} | {label:>8}")


# ═══════════════════════════════════════════════════════════════════
# SECTION 7 — VERIFY MANUAL GRADIENTS MATCH PYTORCH
# The ultimate test — if they match, you understand backprop completely
# ═══════════════════════════════════════════════════════════════════
print("\n\n" + "═" * 60)
print("SECTION 7 — VERIFY MANUAL GRADIENTS MATCH PYTORCH")
print("═" * 60)

print("\nThis is the ultimate test.")
print("If manual gradients == PyTorch gradients, you understand backprop completely.\n")

# Reset both networks to same starting weights
np.random.seed(99)
torch.manual_seed(99)

# Manual network
W1_v = np.random.randn(2, 4) * 0.1
b1_v = np.zeros((1, 4))
W2_v = np.random.randn(4, 1) * 0.1
b2_v = np.zeros((1, 1))

# PyTorch network with SAME weights
model_v = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
    nn.Sigmoid()
)

# Manually set PyTorch weights to match our numpy weights
with torch.no_grad():
    model_v[0].weight.data = torch.tensor(W1_v.T, dtype=torch.float32)
    model_v[0].bias.data   = torch.tensor(b1_v[0], dtype=torch.float32)
    model_v[2].weight.data = torch.tensor(W2_v.T, dtype=torch.float32)
    model_v[2].bias.data   = torch.tensor(b2_v[0], dtype=torch.float32)

# One forward + backward pass — manual
A2_v, cache_v = forward_pass(X, W1_v, b1_v, W2_v, b2_v)
dW1_v, db1_v, dW2_v, db2_v = backward_pass(cache_v, y, W2_v)

# One forward + backward pass — PyTorch
predictions_v = model_v(X_t)
loss_v = nn.BCELoss()(predictions_v, y_t)
model_v.zero_grad()
loss_v.backward()

# Extract PyTorch gradients
dW1_torch = model_v[0].weight.grad.numpy().T
db1_torch = model_v[0].bias.grad.numpy().reshape(1, -1)
dW2_torch = model_v[2].weight.grad.numpy().T
db2_torch = model_v[2].bias.grad.numpy().reshape(1, -1)

# Compare
def compare_gradients(name, manual, pytorch):
    max_diff = np.max(np.abs(manual - pytorch))
    match    = "✅ MATCH" if max_diff < 1e-5 else "❌ MISMATCH"
    print(f"  {name:>8}: max difference = {max_diff:.2e}  {match}")

print("Comparing manual vs PyTorch gradients:")
print("-" * 50)
compare_gradients("dW1", dW1_v, dW1_torch)
compare_gradients("db1", db1_v, db1_torch)
compare_gradients("dW2", dW2_v, dW2_torch)
compare_gradients("db2", db2_v, db2_torch)

print("\nDetailed dW1 comparison:")
print(f"  Manual  dW1:\n{dW1_v}")
print(f"\n  PyTorch dW1:\n{dW1_torch}")

print("\nDetailed dW2 comparison:")
print(f"  Manual  dW2:\n{dW2_v}")
print(f"\n  PyTorch dW2:\n{dW2_torch}")


# ═══════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════
print("\n\n" + "═" * 60)
print("FINAL SUMMARY — WHAT YOU JUST LEARNED")
print("═" * 60)
print("""
1. Backpropagation = chain rule applied backward through layers
   dL/dW = dL/dA * dA/dZ * dZ/dW  (at every layer)

2. Every training step has 4 parts:
   Forward pass  → get predictions
   Loss          → measure how wrong
   Backward pass → compute gradients (backprop)
   Update        → adjust weights (gradient descent)

3. Activation functions matter:
   Sigmoid in hidden layers → vanishing gradients
   ReLU in hidden layers   → gradients flow freely

4. Initialisation matters:
   Zeros → symmetry problem (all neurons learn same thing)
   Random small values → breaks symmetry → network learns

5. Learning rate matters:
   Too high → loss explodes
   Too low  → training too slow
   Just right → steady convergence

6. PyTorch's loss.backward() does exactly what you wrote
   manually in backward_pass() — just faster and for any
   network architecture automatically.

7. If your manual gradients match PyTorch gradients exactly
   (as verified in Section 7) — you understand backprop completely.

Next step: Watch Karpathy's Video 1 — he builds the same thing
from scratch and explains every line in detail.
Link: https://www.youtube.com/watch?v=VMj-3S1tku0
""")

print("=" * 60)
print("Run complete. Study each section carefully.")
print("Delete the backward_pass function and rewrite it from memory.")
print("That is the exercise that cements backpropagation forever.")
print("=" * 60)