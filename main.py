"""
interpretable_nn_analysis.py

This script trains an artificial neural network on an extended Iris dataset,
selects representative examples, generates perturbed versions of those examples,
trains local interpretable linear models on the perturbed data, and visualizes
various metrics and interpretability analyses.

Sections:
1. Data loading and preprocessing (manual One-Hot encoding and scaling)
2. Neural network training
3. Example selection and perturbation
4. Local linear model training
5. Visualization:
   - Feature contributions for each example
   - Perturbation stability across examples
   - Batch predictions
   - Confusion matrix
   - Train vs. validation accuracy
   - Diversity of perturbed predictions
   - Local vs. NN accuracy comparison
   - Feature contribution comparison at ±10% vs ±20% noise
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from NeuralNet import NeuralNet
from Local_model import LinearLocalModel

# ------------------------------------------------------------------------------
# 1. Data Loading and Preprocessing
# ------------------------------------------------------------------------------

# Load extended Iris dataset
df = pd.read_csv("data/iris_extended.csv")

# Manual One-Hot encoding for soil_type
soil_types = sorted(df['soil_type'].unique())
for soil in soil_types:
    df[f"soil_{soil}"] = (df['soil_type'] == soil).astype(float)
df.drop(columns=['soil_type'], inplace=True)

# Extract feature names (exclude the 'species' target column)
feature_names = [col for col in df.columns if col != 'species']
X_raw = df[feature_names].values.astype(float)

# Manual StandardScaler: center to zero mean, unit variance
means = X_raw.mean(axis=0)
stds  = X_raw.std(axis=0)
stds[stds == 0] = 1.0  # avoid division by zero
X = (X_raw - means) / stds

# Manual One-Hot encoding for 'species' target
species = sorted(df['species'].unique())
label_map = {label: idx for idx, label in enumerate(species)}
y = np.zeros((len(df), len(species)))
for i, label in enumerate(df['species']):
    y[i, label_map[label]] = 1.0

# Split into train / validation / test sets
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.55, stratify=y_temp, random_state=42
)

# ------------------------------------------------------------------------------
# 2. Neural Network Training
# ------------------------------------------------------------------------------

# Initialize and train the neural network
nn = NeuralNet(
    hidden_layer_sizes=(16, 8),
    batch_size=20,
    early_stopping=True,
    patience=10,
    activation='tanh',
    learning_rate=0.01,
    epoch=200
)
nn.fit(X_train, y_train)

# ------------------------------------------------------------------------------
# 3. Example Selection and Perturbation
# ------------------------------------------------------------------------------

# Predict on validation set and select 3 correctly and 3 incorrectly classified examples
y_val_pred = np.argmax(nn.predict(X_val), axis=1)
y_val_true = np.argmax(y_val, axis=1)
correct   = np.where(y_val_pred == y_val_true)[0][:3]
incorrect = np.where(y_val_pred != y_val_true)[0][:3]

# Function to generate perturbed samples by adding up to ±10% uniform noise
def generate_perturbed_samples(instance: np.ndarray,
                               n_samples: int = 250,
                               noise_level: float = 0.1) -> np.ndarray:
    """
    Generate perturbed versions of a single data instance.

    Args:
        instance: 1D array of feature values.
        n_samples: Number of perturbed samples to generate.
        noise_level: Maximum relative noise amplitude (e.g., 0.1 for ±10%).

    Returns:
        perturbed_data: Array of shape (n_samples, n_features).
    """
    noise = 1 + noise_level * np.random.uniform(-1, 1, size=(n_samples, instance.size))
    return instance * noise

# Create dictionary of original and perturbed instances + their predicted classes
perturbed_data = {}
for idx in list(correct) + list(incorrect):
    original = X_val[idx]
    perturbed = generate_perturbed_samples(original)
    preds = np.argmax(nn.predict(perturbed), axis=1)
    perturbed_data[idx] = {
        "original": original,
        "perturbed": perturbed,
        "predictions": preds
    }

# ------------------------------------------------------------------------------
# 4. Local Linear Model Training
# ------------------------------------------------------------------------------

# Train a local interpretable linear model (via softmax regression) for each example
local_models = {}
for idx, data in perturbed_data.items():
    X_loc = data["perturbed"]
    y_loc = np.eye(len(species))[data["predictions"]]
    lm = LinearLocalModel(input_dim=X_loc.shape[1])
    lm.fit(X_loc, y_loc)
    local_models[idx] = lm

# ------------------------------------------------------------------------------
# 5. Visualization of Metrics and Interpretations
# ------------------------------------------------------------------------------

# -- 5.1. Feature Contributions Grid --
def plot_feature_importance_grid(local_models: dict, feature_names: list):
    """
    Plot horizontal bar charts of absolute feature contributions for each instance
    in a grid layout (3 columns).

    Args:
        local_models: Dict mapping instance index to trained LinearLocalModel.
        feature_names: List of feature names.
    """
    num = len(local_models)
    cols = 3
    rows = (num + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()

    for ax, (idx, lm) in zip(axes, local_models.items()):
        weights = lm.get_weights()
        contributions = np.abs(weights).sum(axis=1)
        order = np.argsort(contributions)[::-1]
        labels = [feature_names[i] for i in order]
        values = contributions[order]

        y_pos = np.arange(len(labels))
        ax.barh(y_pos, values)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=6)
        ax.invert_yaxis()
        ax.set_title(f'Instance {idx}', fontsize=10)

    # Remove unused subplots
    for ax in axes[num:]:
        fig.delaxes(ax)

    fig.tight_layout()
    plt.xlabel("Absolute Contribution")
    plt.show()

plot_feature_importance_grid(local_models, feature_names)

# -- 5.2. Perturbation Stability Across Instances --
instances = list(perturbed_data.keys())
stabilities = [
    np.mean(perturbed_data[idx]['predictions'] ==
            np.argmax(nn.predict(perturbed_data[idx]['original'].reshape(1, -1)), axis=1)[0])
    for idx in instances
]

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar([str(idx) for idx in instances], stabilities, color='#C44E52')
for bar, val in zip(bars, stabilities):
    ax.text(bar.get_x() + bar.get_width() / 2,
            val + 0.02,
            f'{val*100:.0f}%',
            ha='center', va='bottom', fontsize=9)

ax.set_ylim(0, 1.1)
ax.set_xlabel('Instance')
ax.set_ylabel('Fraction of perturbed samples\nwith same NN prediction')
ax.set_title('Perturbation Stability Across Instances', pad=20, fontsize=14)
plt.tight_layout()
plt.show()

# -- 5.3. Batch Predictions for 4 Test Examples --
batch_idx   = np.random.choice(len(X_test), size=4, replace=False)
batch_probs = nn.predict(X_test[batch_idx])
batch_true  = np.argmax(y_test[batch_idx], axis=1)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, idx, probs, true in zip(axes, batch_idx, batch_probs, batch_true):
    ax.bar(np.arange(len(species)), probs)
    ax.set_xticks(np.arange(len(species)))
    ax.set_xticklabels(species, rotation=45, ha='right')
    ax.set_ylim(0, 1)
    ax.set_title(f'Inst {idx}\nTrue: {species[true]}')
    ax.set_ylabel('Probability')

plt.tight_layout()
plt.show()

# -- 5.4. Confusion Matrix on Test Set --
y_test_pred = np.argmax(nn.predict(X_test), axis=1)
y_test_true = np.argmax(y_test, axis=1)
conf = confusion_matrix(y_test_true, y_test_pred)

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(conf, cmap='Blues')
for i in range(conf.shape[0]):
    for j in range(conf.shape[1]):
        ax.text(j, i, conf[i, j],
                ha='center', va='center',
                color='white' if conf[i, j] > conf.max()/2 else 'black')

ax.set_xticks(np.arange(len(species)))
ax.set_yticks(np.arange(len(species)))
ax.set_xticklabels(species, rotation=45, ha='right')
ax.set_yticklabels(species)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Test Set Confusion Matrix')
fig.colorbar(im, ax=ax)
plt.tight_layout()
plt.show()

# -- 5.5. Train vs. Validation Accuracy --
train_acc = accuracy_score(np.argmax(y_train, axis=1), np.argmax(nn.predict(X_train), axis=1))
val_acc   = accuracy_score(np.argmax(y_val,   axis=1), np.argmax(nn.predict(X_val),   axis=1))

fig, ax = plt.subplots(figsize=(5, 3))
ax.bar(['Train', 'Val'], [train_acc, val_acc], color=['#4C72B0', '#55A868'])
ax.set_ylim(0, 1)
ax.set_ylabel('Accuracy')
ax.set_title('Train vs. Validation Accuracy')
for i, val in enumerate([train_acc, val_acc]):
    ax.text(i, val + 0.02, f'{val:.3f}', ha='center')

plt.tight_layout()
plt.show()

# -- 5.6. Diversity of Perturbed Predictions --
counts = [len(np.unique(perturbed_data[idx]['predictions'])) for idx in instances]

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar([str(idx) for idx in instances], counts, color='#8172B2')
ax.set_xlabel('Instance')
ax.set_ylabel('Number of unique predicted classes')
ax.set_title('Diversity of Perturbed Predictions')
for i, c in enumerate(counts):
    ax.text(i, c + 0.1, str(c), ha='center')

plt.tight_layout()
plt.show()

# -- 5.7. NN self-consistency vs. Local model vs. Ground-truth Accuracy Across Instances --
instances = list(perturbed_data.keys())
fidelity = []

for idx in instances:
    # NN predictions on perturbed samples (already stored)
    nn_preds = perturbed_data[idx]['predictions']
    # Local model predictions
    local_preds = np.argmax(
        local_models[idx].predict_proba(perturbed_data[idx]['perturbed']),
        axis=1
    )
    # Fidelity = fraction of matching predictions
    fidelity.append(np.mean(local_preds == nn_preds))

# Plot fidelity
fig, ax = plt.subplots(figsize=(8,4))
bars = ax.bar([str(i) for i in instances], fidelity, color='#348ABD')

# Annotate percentage on top
for bar, val in zip(bars, fidelity):
    ax.text(
        bar.get_x() + bar.get_width()/2,
        val + 0.02,
        f'{val*100:.1f}%',
        ha='center',
        va='bottom'
    )

ax.set_ylim(0, 1.1)
ax.set_xlabel('Instance')
ax.set_ylabel('Fidelity (Local reproduces NN)')
ax.set_title(
    'Local Model Fidelity Across Instances',
    fontsize=12,
    pad=15
)
plt.tight_layout()
plt.show()

# -- 5.8. Feature Contribution Comparison ±10% vs ±20% (Instance 0) --
idx0 = correct[0]  # first correctly classified instance
model0 = local_models[idx0]

# Contributions at ±10%
w10 = np.abs(model0.get_weights()).sum(axis=1)

# Contributions at ±20%
pert20 = generate_perturbed_samples(perturbed_data[idx0]['original'], noise_level=0.2)
y20    = np.argmax(nn.predict(pert20), axis=1)
model20 = LinearLocalModel(input_dim=X.shape[1])
model20.fit(pert20, np.eye(len(species))[y20])
w20 = np.abs(model20.get_weights()).sum(axis=1)

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(feature_names))
width = 0.4
ax.bar(x - width/2, w10, width, label='±10%')
ax.bar(x + width/2, w20, width, label='±20%')
ax.set_xticks(x)
ax.set_xticklabels(feature_names, rotation=90, ha='center')
ax.set_ylabel('Absolute Contribution')
ax.set_title(f'Feature Contributions for Instance {idx0}: ±10% vs ±20%')
ax.legend()

plt.tight_layout()
plt.show()

