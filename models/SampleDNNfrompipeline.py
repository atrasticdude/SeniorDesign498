import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
import tensorflow as tf
from sklearn.metrics import roc_curve, auc, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt

df = pd.read_csv("Z:/Users/Artin/Data/01_SPSS.csv", header=None)
# X_raw = df.iloc[1:, 42:62].astype(float).values
# Y_raw = df.iloc[1:, 1].astype(float).values
# filenames = df.iloc[1:, 0]
# labels = df.iloc[1:, 1].astype(float).values
# column_order = [
#     "PH_NR_mean",
#     "TTP_NR_mean",
#     "AUC_NR_mean",
#     "MTT_NR_mean",
#     "max_Df_NR_mean",
#     "AUC_05MTT_NR_mean",
#     "AUC_1MTT_NR_mean",
#     "AUC_15MTT_NR_mean",
#     "AUC_2MTT_NR_mean"
# ]
column_order = [
    "PH_NR_mean",
    "AUC_NR_mean",
    "max_Df_NR_mean"
]
df.columns = df.iloc[0]
df = df[1:]
X_raw = df[column_order].iloc[1:].astype(float).values
Y_raw = df.iloc[1:, 1].astype(float).values
filenames = df.iloc[1:, 0]
labels = df.iloc[1:, 1].astype(float).values


def get_base_id(name):
    return "_".join(name.split("_")[:2])

base_ids = [get_base_id(f) for f in filenames]
grouped_files = defaultdict(list)
grouped_labels = {}
for f, l in zip(filenames, labels):
    base = get_base_id(f)
    grouped_files[base].append(f)
    grouped_labels[base] = max(grouped_labels.get(base, 0), l)

base_id_list = list(grouped_files.keys())
base_labels = [grouped_labels[bid] for bid in base_id_list]

X_features = X_raw
z_scores = np.abs((X_features - X_features.mean(axis=0)) / X_features.std(axis=0))
mask = (z_scores < 3).all(axis=1)
X_raw = X_raw[mask]
Y_raw = Y_raw[mask]
filenames = filenames[mask]

filename_to_index = {name: idx for idx, name in enumerate(filenames)}

sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=998)
train_files = defaultdict(list)
val_files = defaultdict(list)

for fold, (train_idx, val_idx) in enumerate(sss.split(base_id_list, base_labels)):
    train_bases = [base_id_list[i] for i in train_idx]
    val_bases = [base_id_list[i] for i in val_idx]
    train_files[fold].append([f for b in train_bases for f in grouped_files[b]])
    val_files[fold].append([f for b in val_bases for f in grouped_files[b]])

all_results = []

for fold, file_lists in enumerate(tqdm(train_files, desc="Training Folds")):
    train_filenames = train_files[fold][0]
    val_filenames = val_files[fold][0]

    train_indices = [filename_to_index[f] for f in train_filenames if f in filename_to_index]
    val_indices = [filename_to_index[f] for f in val_filenames if f in filename_to_index]

    X_train = X_raw[train_indices]
    y_train = Y_raw[train_indices]
    X_val = X_raw[val_indices]
    y_val = Y_raw[val_indices]

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    ros = RandomOverSampler(random_state=fold)
    X_train, y_train = ros.fit_resample(X_train, y_train)

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    filepath = f'best_weights_fold{fold+1}.h5'
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    check_point = tf.keras.callbacks.ModelCheckpoint(filepath=filepath, monitor='val_loss', mode='min', save_best_only=True, verbose=1)

    model.compile(optimizer=tf.keras.optimizers.Adam(0.0005),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=32,
        epochs=100,
        callbacks=[early_stopping, check_point],
        verbose=1
    )

    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy (Fold {fold+1})')
    plt.legend(['Train', 'Val'])
    plt.show()

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss (Fold {fold+1})')
    plt.legend(['Train', 'Val'])
    plt.show()

    y_prob = model.predict(X_val).ravel()
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'Fold {fold+1} AUROC = {roc_auc:.3f}')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'AUROC Curve (Fold {fold+1})')
    plt.legend(loc='lower right')
    plt.show()

    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_val, y_pred)

    all_results.append({
        "fold": fold+1,
        "train_acc": history.history["accuracy"][-1],
        "val_acc": history.history["val_accuracy"][-1],
        "roc_auc": roc_auc,
        "confusion_matrix": cm
    })

avg_roc = np.mean([r["roc_auc"] for r in all_results])
avg_val_acc = np.mean([r["val_acc"] for r in all_results])

print("Average ROC AUC:", avg_roc)
print("Average Val Accuracy:", avg_val_acc)

last = all_results[-1]
plt.figure()
plt.imshow(last["confusion_matrix"], cmap='Blues')
plt.colorbar()
plt.xticks([0,1], ['Pred 0','Pred 1'])
plt.yticks([0,1], ['True 0','True 1'])
for i in range(2):
    for j in range(2):
        plt.text(j, i, last["confusion_matrix"][i,j], ha='center', va='center')
plt.title("Confusion Matrix (Last Fold)")
plt.show()
