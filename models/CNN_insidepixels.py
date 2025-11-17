


from pathlib import Path
import os
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, Subset
import matplotlib.pyplot as plt


def load_data(directory, file):
    csv_path = Path(__file__).resolve().parent.parent / directory / file
    df = pd.read_csv(csv_path)
    return df

inclusion_df= load_data("Data", "cnn_inclusion.csv")
label_df = load_data("Data", "Coil_study_occlusion_key.csv")

cases_inclusion = inclusion_df["Case Number"].values
cases_label = label_df["Case Number"].values

mask = inclusion_df["Outcome"] == 1
true_cases = cases_inclusion[mask]

label_map = label_df.set_index("Case Number")["Outcome"]
true_labels = label_map.loc[true_cases].values

image_path = r"Z:\Users\Artin\coiled\cnn_pics"

true_data = []
true_labels_filtered = []

for case, label in zip(true_cases, true_labels):
    filename = f"{case}_1.png"
    full_path = os.path.join(image_path, filename)

    if os.path.exists(full_path):
        img = cv2.imread(full_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        true_data.append(img)
        true_labels_filtered.append(label)
    else:
        print(f"Missing file: {full_path}")

data_with_labels = list(zip(true_labels_filtered, true_data))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

# === Convert data to tensors ===

# === Prepare data ===
X = np.array([img for _, img in data_with_labels], dtype=np.float32) / 255.0
y = np.array([label for label, _ in data_with_labels], dtype=np.float32)

X_tensor = torch.tensor(np.transpose(X, (0, 3, 1, 2)))
y_tensor = torch.tensor(y).unsqueeze(1)  # shape (N, 1)

dataset = TensorDataset(X_tensor, y_tensor)

# === Split into train + test once ===
train_idx, test_idx = train_test_split(np.arange(len(dataset)), test_size=0.2, random_state=42)
train_set_full = Subset(dataset, train_idx)
test_set = Subset(dataset, test_idx)

# === Monte Carlo split function for train/validation ===
def monte_carlo_train_val_splits(train_set, val_size=0.2, iterations=5, batch_size=16):
    splits = []
    n_total = len(train_set)
    n_val = int(val_size * n_total)

    for _ in range(iterations):
        indices = torch.randperm(n_total)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        train_subset = Subset(train_set, train_indices)
        val_subset = Subset(train_set, val_indices)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        splits.append((train_loader, val_loader))
    return splits

# === CNN ===
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * (X.shape[1] // 8) * (X.shape[2] // 8), 128)
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
        x = torch.sigmoid(self.fc_out(x))
        return x

# === Training ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
num_epochs = 20
batch_size = 16
val_size = 0.2
iterations = 5

# Generate Monte Carlo train/val splits
mc_splits = monte_carlo_train_val_splits(train_set_full, val_size=val_size, iterations=iterations, batch_size=batch_size)

# Use first Monte Carlo split for training here (example)
train_loader, val_loader = mc_splits[0]

train_losses, val_losses = [], []
train_accs, val_accs = [], []

plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for epoch in range(num_epochs):
    # === Training ===
    model.train()
    running_loss = 0
    correct = 0
    total = 0

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
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # === Validation ===
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
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

    print(f"Epoch {epoch + 1}/{num_epochs} | "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # === Update plots after each epoch ===
    ax1.clear()
    ax1.plot(range(1, epoch + 2), train_losses, label='Train Loss')
    ax1.plot(range(1, epoch + 2), val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss over Epochs')
    ax1.legend()

    ax2.clear()
    ax2.plot(range(1, epoch + 2), train_accs, label='Train Accuracy')
    ax2.plot(range(1, epoch + 2), val_accs, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy over Epochs')
    ax2.legend()

    plt.pause(0.1)

plt.ioff()
plt.show()

from sklearn.metrics import confusion_matrix
import seaborn as sns

# === After validation loop in each epoch ===
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = (outputs > 0.5).float()
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Validation Confusion Matrix - Epoch {epoch+1}')
plt.show()


#
# import  os
# from pathlib import Path
# import cv2
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader, Subset
# from sklearn.model_selection import train_test_split
# from torchvision import transforms
# import matplotlib.pyplot as plt
#
# # =========================
# # 1. Data Loading
# # =========================
# def load_data(directory, file):
#     csv_path = Path(__file__).resolve().parent.parent / directory / file
#     df = pd.read_csv(csv_path)
#     return df
#
# inclusion_df = load_data("Data", "cnn_inclusion.csv")
# label_df = load_data("Data", "Coil_study_occlusion_key.csv")
#
# cases_inclusion = inclusion_df["Case Number"].values
# mask = inclusion_df["Outcome"] == 1
# true_cases = cases_inclusion[mask]
#
# # Map labels from label_df
# label_map = label_df.set_index("Case Number")["Outcome"]
# true_cases = [case for case in true_cases if case in label_map]
# true_labels = label_map.loc[true_cases].values
#
# image_path = r"Z:\Users\Artin\coiled\cnn_pics"
#
# # =========================
# # 2. Custom Dataset with Augmentation & Missing File Handling
# # =========================
# class AugmentedImageDataset(Dataset):
#     def __init__(self, cases, labels, img_dir, transform=None):
#         self.img_dir = img_dir
#         self.transform = transform
#
#         # Filter out missing/corrupted images at initialization
#         valid_cases = []
#         valid_labels = []
#         for case, label in zip(cases, labels):
#             filename = f"{case}_1.png"
#             full_path = os.path.join(img_dir, filename)
#             img = cv2.imread(full_path)
#             if img is not None:
#                 valid_cases.append(case)
#                 valid_labels.append(label)
#             else:
#                 print(f"Skipping missing/corrupted image: {full_path}")
#
#         self.cases = valid_cases
#         self.labels = valid_labels
#
#     def __len__(self):
#         return len(self.cases)
#
#     def __getitem__(self, idx):
#         case = self.cases[idx]
#         label = self.labels[idx]
#         filename = f"{case}_1.png"
#         full_path = os.path.join(self.img_dir, filename)
#         img = cv2.imread(full_path, cv2.IMREAD_COLOR)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#         if self.transform:
#             img = self.transform(img)
#         else:
#             img = torch.from_numpy(np.transpose(img / 255.0, (2,0,1))).float()
#
#         label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
#         return img, label
#
# # --- Augmentation for training ---
# train_transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(10),
#     transforms.ColorJitter(brightness=0.1, contrast=0.1),
#     transforms.ToTensor()
# ])
#
# # --- No augmentation for validation/test ---
# val_transform = transforms.Compose([
#     transforms.ToTensor()
# ])
#
# # =========================
# # 3. Prepare Datasets
# # =========================
# full_dataset = AugmentedImageDataset(true_cases, true_labels, image_path, transform=None)
#
# # Split indices
# train_idx, test_idx = train_test_split(range(len(full_dataset)), test_size=0.2, random_state=42)
# train_dataset = Subset(full_dataset, train_idx)
# test_dataset = Subset(full_dataset, test_idx)
#
# # Assign transforms
# train_dataset.dataset.transform = train_transform
# test_dataset.dataset.transform = val_transform
#
# batch_size = 8
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#
# # =========================
# # 4. CNN Model with Adaptive Pooling
# # =========================
# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3,16,3,padding=1)
#         self.conv2 = nn.Conv2d(16,32,3,padding=1)
#         self.conv3 = nn.Conv2d(32,64,3,padding=1)
#         self.pool = nn.MaxPool2d(2)
#         self.adaptive_pool = nn.AdaptiveAvgPool2d((4,4))
#         self.dropout = nn.Dropout(0.5)
#         self.fc1 = nn.Linear(64*4*4,64)
#         self.fc2 = nn.Linear(64,32)
#         self.fc_out = nn.Linear(32,1)
#
#     def forward(self, x):
#         x = self.pool(torch.relu(self.conv1(x)))
#         x = self.pool(torch.relu(self.conv2(x)))
#         x = self.pool(torch.relu(self.conv3(x)))
#         x = self.adaptive_pool(x)
#         x = torch.flatten(x,1)
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.dropout(x)
#         x = self.fc_out(x)
#         return x
#
# # =========================
# # 5. Training Setup
# # =========================
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = SimpleCNN().to(device)
#
# criterion = nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
#
# num_epochs = 50
# patience = 5  # early stopping
# best_val_loss = float('inf')
# epochs_no_improve = 0
#
# train_losses, val_losses = [], []
#
# # =========================
# # 6. Training Loop
# # =========================
# for epoch in range(num_epochs):
#     # --- Training ---
#     model.train()
#     running_loss = 0
#     for imgs, labels in train_loader:
#         imgs, labels = imgs.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(imgs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item() * imgs.size(0)
#     train_loss = running_loss / len(train_loader.dataset)
#     train_losses.append(train_loss)
#
#     # --- Validation ---
#     model.eval()
#     val_loss = 0
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for imgs, labels in val_loader:
#             imgs, labels = imgs.to(device), labels.to(device)
#             outputs = model(imgs)
#             loss = criterion(outputs, labels)
#             val_loss += loss.item() * imgs.size(0)
#
#             preds = (torch.sigmoid(outputs) > 0.5).float()
#             correct += (preds == labels).sum().item()
#             total += labels.size(0)
#
#     val_loss /= len(val_loader.dataset)
#     val_acc = correct / total
#     val_losses.append(val_loss)
#
#     print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | "
#           f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
#
#     # --- Early stopping ---
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         epochs_no_improve = 0
#         torch.save(model.state_dict(), "best_model.pth")
#     else:
#         epochs_no_improve += 1
#         if epochs_no_improve >= patience:
#             print("Early stopping triggered")
#             break
#
# # =========================
# # 7. Load Best Model
# # =========================
# model.load_state_dict(torch.load("best_model.pth"))
#
# # =========================
# # 8. Plot Losses
# # =========================
# plt.figure(figsize=(8,5))
# plt.plot(train_losses, label='Train Loss')
# plt.plot(val_losses, label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training vs Validation Loss')
# plt.legend()
# plt.show()
# from sklearn.metrics import confusion_matrix
# import seaborn as sns
#
# # === After validation loop in each epoch ===
# y_true, y_pred = [], []
#
# with torch.no_grad():
#     for images, labels in val_loader:
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images)
#         preds = (outputs > 0.5).float()
#         y_true.extend(labels.cpu().numpy())
#         y_pred.extend(preds.cpu().numpy())
#
# # Compute confusion matrix
# cm = confusion_matrix(y_true, y_pred)
#
# # Plot confusion matrix
# plt.figure(figsize=(5, 4))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title(f'Validation Confusion Matrix - Epoch {epoch+1}')
# plt.show()



