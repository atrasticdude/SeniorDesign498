
import torch
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from collections import defaultdict
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import shuffle
from imblearn.over_sampling import RandomOverSampler
import tensorflow as tf
import re
from sklearn.decomposition import PCA
import matplotlib.pyplot as pl


df = pd.read_csv("Z:/Users/Artin/Data/01_SPSS.csv", header=None)
X_raw = df.iloc[1:, 42:62].astype(float).values
Y_raw = df.iloc[1:, 1].astype(float).values

filenames = df.iloc[1:,0]
labels = df.iloc[1:,1]




def get_base_id(name):
    return "_".join(name.split("_")[:2]) 

base_ids = [get_base_id(f) for f in filenames]


grouped_files = defaultdict(list)
grouped_labels = {}

for f, l in zip(filenames, labels):
    base = get_base_id(f)
    grouped_files[base].append(f)
    grouped_labels[base] = l 


base_id_list = list(grouped_files.keys())
base_labels = [grouped_labels[bid] for bid in base_id_list]


sss = StratifiedShuffleSplit(n_splits=20, test_size=0.2, random_state=998)

train_files = defaultdict(list)
val_files = defaultdict(list)

for fold, (train_idx, val_idx) in enumerate(sss.split(base_id_list, base_labels)):
    train_bases = [base_id_list[i] for i in train_idx]
    val_bases = [base_id_list[i] for i in val_idx]

    train_files[fold].append([f for b in train_bases for f in grouped_files[b]])
    val_files[fold].append([f for b in val_bases for f in grouped_files[b]])

    # Placeholder: Load your data for this fold
    # train_data, train_labels = load_data(train_files)
    # val_data, val_labels = load_data(val_files)


val_files_items = list(val_files.items())
mid = len(val_files_items) // 2
val_files_split = dict(val_files_items[:mid])
test_files = dict(val_files_items[mid:])


filename_to_index = {name: idx for idx, name in enumerate(filenames)}

for fold, file_lists in train_files.items():
    train_filenames = file_lists[0]  
    val_filenames = val_files[fold][0]

    train_indices = [filename_to_index[f] for f in train_filenames]
    val_indices = [filename_to_index[f] for f in val_filenames]

    X_train = X_raw[train_indices]
    y_train = Y_raw[train_indices]
    X_val = X_raw[val_indices]
    y_val = Y_raw[val_indices]


    ros = RandomOverSampler(random_state=fold)
    X_train, y_train = ros.fit_resample(X_train, y_train)

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    filepath = f'best_weights_fold{fold+1}.h5'
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    check_point = tf.keras.callbacks.ModelCheckpoint(
        filepath=filepath, monitor='val_loss', mode='min', save_best_only=True, verbose=1
    )

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=32,
        epochs=100,
        callbacks=[early_stopping, check_point]
    )

    
