import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

CLASS_NAMES = ["No Finding", "Pneumonia", "Other Diseases"]
mlb = MultiLabelBinarizer(classes=CLASS_NAMES)

# Load CSV
train_df = pd.read_csv("Xray Image Classification/image_labels_train.csv")

# Define the labels you're interested in
CLASS_NAMES = ["No finding", "Pneumonia", "Other disease"]

# Keep only image_id and those class columns
train_df = train_df[["image_id"] + CLASS_NAMES]

# Remove rows where all class labels are 0 (i.e., no relevant condition present)
train_df = train_df[train_df[CLASS_NAMES].sum(axis=1) > 0]

# No need to encode anything – the labels are already in multi-hot format
# You can directly use: train_df[CLASS_NAMES].values for labels
# Oversample Pneumonia and Other disease
minority_df = train_df[(train_df["Pneumonia"] == 1) | (train_df["Other disease"] == 1)]
train_df = pd.concat([train_df, minority_df, minority_df], ignore_index=True).sample(frac=1).reset_index(drop=True)

path='Xray Image Classification/train'

def load_dicom_image(path, img_size=(224, 224)): 
    dicom = pydicom.dcmread(path)
    img = apply_voi_lut(dicom.pixel_array, dicom)
    img = img.astype(np.float32)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))  # Normalize to [0, 1]
    img = np.stack([img] * 3, axis=-1)  # Convert to 3-channel image
    img = tf.image.resize(img, img_size)

    img = preprocess_input(img)  # 
    return img


def create_dataset(df, base_dir, batch_size=8, shuffle=True):
    paths = df['image_id'].apply(lambda x: os.path.join(base_dir, f"{x}.dicom")).values
    labels = df[CLASS_NAMES].values.astype(np.float32)  # ✅ Use actual class columns

    def gen():
        for path, label in zip(paths, labels):
            try:
                img = load_dicom_image(path)
                yield img, label
            except:
                continue  # Skip unreadable or missing files

    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(len(CLASS_NAMES),), dtype=tf.float32)
        )
    )
    if shuffle:
        dataset = dataset.shuffle(1024)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

from sklearn.utils import resample

# Step 1: Extract the subsets (multi-label allowed)
no_finding_df = train_df[train_df["No finding"] == 1]
pneumonia_df = train_df[train_df["Pneumonia"] == 1]
other_disease_df = train_df[train_df["Other disease"] == 1]

# Step 2: Oversample PNEUMONIA and OTHER only — DON'T touch No Finding
target_count = max(len(pneumonia_df), len(other_disease_df))

pneumonia_upsampled = resample(pneumonia_df, 
                                replace=True, 
                                n_samples=target_count, 
                                random_state=42)

other_upsampled = resample(other_disease_df, 
                           replace=True, 
                           n_samples=target_count, 
                           random_state=42)

# Step 3: Combine everything (oversampled + original no finding)
balanced_df = pd.concat([no_finding_df, pneumonia_upsampled, other_upsampled])
balanced_df = balanced_df.drop_duplicates(subset='image_id')  # remove exact duplicates
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

# Step 4: Show new distribution
print("✅ Final balanced image count per class:")
print(balanced_df[["No finding", "Pneumonia", "Other disease"]].sum().astype(int))

target_count = max(len(no_finding_df), len(pneumonia_df), len(other_disease_df))
# Then resample all 3 to this count

# Convert multi-label binary columns into tuples for stratification
train_df['label_tuple'] = train_df[CLASS_NAMES].apply(lambda row: tuple(row.astype(int)), axis=1)

# Perform the split
train_data, val_data = train_test_split(
    train_df,
    test_size=0.2,
    stratify=train_df['label_tuple'],
    random_state=42
)

# Optional: Drop the helper column
train_data = train_data.drop(columns=["label_tuple"])
val_data = val_data.drop(columns=["label_tuple"])

# Count number of images where each class is labeled as 1
class_counts = train_df[["No finding", "Pneumonia", "Other disease"]].sum().astype(int)

print("✅ Number of images per class:")
print(class_counts)
from focal_loss import SparseCategoricalFocalLoss

def weighted_sigmoid_focal_loss(class_weights, gamma=2.0):
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        loss = - y_true * tf.pow(1 - y_pred, gamma) * tf.math.log(y_pred)
        loss -= (1 - y_true) * tf.pow(y_pred, gamma) * tf.math.log(1 - y_pred)
        
        weights = tf.constant([class_weights[i] for i in range(len(class_weights))], dtype=tf.float32)
        loss *= weights
        return tf.reduce_mean(loss)
    return loss_fn
class_totals = train_df[["No finding", "Pneumonia", "Other disease"]].sum()
total_samples = len(train_df)
class_weights = {
    0: total_samples / (3 * class_totals["No finding"]),
    1: total_samples / (3 * class_totals["Pneumonia"]),
    2: total_samples / (3 * class_totals["Other disease"]),
}


print("Class weights:", class_weights)


base_model = ResNet101(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False
x = base_model.output

x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
output = Dense(len(CLASS_NAMES), activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=output)



model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
   loss=weighted_sigmoid_focal_loss(class_weights, gamma=2.0),

    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn')
    ]
)


model.summary()

train_dataset = create_dataset(train_data, base_dir=path)
val_dataset = create_dataset(val_data, base_dir=path, shuffle=False)

batch_size = 8  # or whatever you're using

train_dataset = train_dataset.repeat()
val_dataset = val_dataset.repeat()
# train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
# val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

steps_per_epoch = len(train_data) // batch_size
validation_steps = len(val_data) // batch_size

callbacks = [
    EarlyStopping(monitor="val_auc", patience=5, mode="max", restore_best_weights=True),
    ModelCheckpoint("best_model.keras", save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_auc',
    factor=0.5,
    patience=2,
    min_lr=1e-7,
    verbose=1
)
]

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=10,
    callbacks=callbacks
)

# 🔓 Unfreeze last 30 layers of base model
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Recompile with a smaller LR


model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-6),
   loss=weighted_sigmoid_focal_loss(class_weights, gamma=2.0),

    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn')
    ]
)



# Continue training
history_finetune_20 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=10,
    callbacks=callbacks
)

# 🔓 Unfreeze all layers
for layer in base_model.layers:
    layer.trainable = True



model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-6),
   loss=weighted_sigmoid_focal_loss(class_weights, gamma=2.0),

    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn')
    ]
)

# Final training phase
history_finetune_all = model.fit(
    train_dataset,
    validation_data=val_dataset,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=10,
    callbacks=callbacks
)
model.save("xray_resnet_model.keras")


import numpy as np
from sklearn.metrics import precision_recall_curve, classification_report, roc_auc_score, hamming_loss, multilabel_confusion_matrix

# CLASS_NAMES used in your model
CLASS_NAMES = ["No finding", "Pneumonia", "Other disease"]

# ------------------------
# Step 1: Collect Predictions and True Labels
# ------------------------
y_true = []
y_pred = []

# Use this to prevent infinite looping due to .repeat()
eval_steps = validation_steps  # same as used during model.fit()

for step, (images, labels) in enumerate(val_dataset):
    if step >= eval_steps:
        break
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(preds)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("✅ Collected predictions and labels")

# ------------------------
# Step 2: Tune optimal thresholds per class using F1
# ------------------------
optimal_thresholds = []

print("\n🔍 Optimal Thresholds (per class):")
for i, class_name in enumerate(CLASS_NAMES):
    precision, recall, thresholds = precision_recall_curve(y_true[:, i], y_pred[:, i])
    f1_scores = 2 * precision * recall / (precision + recall + 1e-6)
    best_thresh = thresholds[np.argmax(f1_scores)]
    optimal_thresholds.append(best_thresh)
    print(f"  {class_name}: Best threshold = {best_thresh:.2f}")

# ------------------------
# Step 3: Apply thresholds and binarize predictions
# ------------------------
y_pred_bin = np.array([
    [1 if p >= t else 0 for p, t in zip(sample, optimal_thresholds)]
    for sample in y_pred
])

# ------------------------
# Step 4: Print Evaluation Metrics
# ------------------------

# Classification report
print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred_bin, target_names=CLASS_NAMES))

# AUC Scores
print("\n📈 AUC Scores (per class):")
for i, class_name in enumerate(CLASS_NAMES):
    auc = roc_auc_score(y_true[:, i], y_pred[:, i])
    print(f"  {class_name}: AUC = {auc:.4f}")

# Hamming Loss
print("\n🧮 Hamming Loss:", hamming_loss(y_true, y_pred_bin))

# # Confusion Matrices
# print("\n🧾 Confusion Matrix (per class):")
# conf_matrices = multilabel_confusion_matrix(y_true, y_pred_bin)
# for i, class_name in enumerate(CLASS_NAMES):
#     print(f"\n{class_name}:\n{conf_matrices[i]}")


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

plt.figure(figsize=(10, 6))

for i, class_name in enumerate(CLASS_NAMES):
    fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Each Class")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix

conf_matrices = multilabel_confusion_matrix(y_true, y_pred_bin)

for i, class_name in enumerate(CLASS_NAMES):
    cm = conf_matrices[i]
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
    plt.title(f"Confusion Matrix: {class_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.show()


per_class_acc = (y_true == y_pred_bin).sum(axis=0) / y_true.shape[0]
for i, class_name in enumerate(CLASS_NAMES):
    print(f"✅ Accuracy for {class_name}: {per_class_acc[i]:.4f}")


overall_label_acc = (y_true == y_pred_bin).mean()
print(f"✅ Overall Label Accuracy: {overall_label_acc:.4f}")
