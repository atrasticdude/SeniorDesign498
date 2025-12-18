from pathlib import Path
import os
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns


def load_data(directory, file):
    csv_path = Path(__file__).resolve().parent.parent / directory / file
    return pd.read_csv(csv_path)

label_df = load_data("Data", "Coil_study_occlusion_key.csv")

cases  = label_df["Case Number"].values
labels = label_df["Outcome"].values.astype(np.float32)


image_path = r"Z:\Users\Artin\coiled\cropped_PCAandmasks"

data = []
labels_filtered = []

for case, label in zip(cases, labels):
    mask_path = os.path.join(image_path, f"{case}_1.tif")
    pca_path  = os.path.join(image_path, f"{case}_1_PCA.tif")

    if not os.path.exists(mask_path):
        print(f"Missing mask: {mask_path}")
        continue
    if not os.path.exists(pca_path):
        print(f"Missing PCA: {pca_path}")
        continue

    mask_img = tifffile.imread(mask_path).astype(np.float32)
    pca_img  = tifffile.imread(pca_path).astype(np.float32)

    if mask_img.size == 0 or pca_img.size == 0:
        print(f"Empty image for case {case}")
        continue

    combined = np.stack([mask_img, pca_img], axis=0)  # (2, H, W)

    data.append(combined)
    labels_filtered.append(label)

X = np.array(data, dtype=np.float32)    # (N, 2, H, W)
y = np.array(labels_filtered, dtype=np.float32)

print("Final dataset shape:", X.shape)

X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y).unsqueeze(1)

dataset = TensorDataset(X_tensor, y_tensor)

train_idx, test_idx = train_test_split(
    np.arange(len(dataset)),
    test_size=0.2,
    random_state=42,
    stratify=y
)

train_set_full = Subset(dataset, train_idx)
test_set       = Subset(dataset, test_idx)


def monte_carlo_train_val_splits(train_set, val_size=0.2, iterations=5, batch_size=16):
    splits = []
    n_total = len(train_set)
    n_val = int(val_size * n_total)

    for _ in range(iterations):
        indices = torch.randperm(n_total)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        train_subset = Subset(train_set, train_indices)
        val_subset   = Subset(train_set, val_indices)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        splits.append((train_loader, val_loader))
    return splits


class SimpleCNN(nn.Module):
    def __init__(self, input_shape):
        super().__init__()

        self.conv1 = nn.Conv2d(2, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            dummy = self.pool(torch.relu(self.conv1(dummy)))
            dummy = self.pool(torch.relu(self.conv2(dummy)))
            dummy = self.pool(torch.relu(self.conv3(dummy)))
            fc_size = dummy.numel()

        self.fc1 = nn.Linear(fc_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc_out = nn.Linear(32, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        return torch.sigmoid(self.fc_out(x))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_shape = X_tensor.shape[1:]  # (2, H, W)
model = SimpleCNN(input_shape).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 20
batch_size = 16
val_size = 0.2
iterations = 5

mc_splits = monte_carlo_train_val_splits(
    train_set_full,
    val_size=val_size,
    iterations=iterations,
    batch_size=batch_size
)

train_loader, val_loader = mc_splits[0]


train_losses, val_losses = [], []
train_accs, val_accs = [], []

plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for epoch in range(num_epochs):

    model.train()
    running_loss, correct, total = 0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = (outputs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc  = correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    model.eval()
    val_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss /= total
    val_acc = correct / total
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(
        f"Epoch {epoch+1}/{num_epochs} | "
        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
    )

    ax1.clear()
    ax1.plot(train_losses, label="Train Loss")
    ax1.plot(val_losses, label="Val Loss")
    ax1.legend()
    ax1.set_title("Loss")

    ax2.clear()
    ax2.plot(train_accs, label="Train Acc")
    ax2.plot(val_accs, label="Val Acc")
    ax2.legend()
    ax2.set_title("Accuracy")

    plt.pause(0.1)

plt.ioff()
plt.show()

y_true, y_pred = [], []

model.eval()
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = (outputs > 0.5).float()

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Validation Confusion Matrix")
plt.show()

