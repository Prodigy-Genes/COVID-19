import os 
import json
import collections
from collections import Counter
from pathlib import Path 
import glob

import numpy as np # for numerical operations
import pandas as pd # for data manipulation
from PIL import Image # for image processing
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for statistical data visualization

import tensorflow as tf # for deep learning
from sklearn.preprocessing import LabelEncoder # for encoding categorical labels
from sklearn.model_selection import train_test_split # for splitting data into training and testing sets
from sklearn.metrics import (
    classification_report, # for generating classification report
    confusion_matrix, # for generating confusion matrix
    roc_curve, roc_auc_score, # for ROC curve and AUC score
    precision_recall_curve, average_precision_score, # for precision-recall curve and average precision score
    accuracy_score, f1_score, # for accuracy and F1 score
)
from sklearn.calibration import calibration_curve
from tensorflow.keras import Sequential, layers # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

# ---- PATHS ----
# Define the paths to the dataset and the model
BASE =  Path(__file__).resolve().parent.parent
DATA_PATH = BASE / "datasets" / "COVID-19"
METRICS = BASE / "metrics"
MODELS = BASE / "model"
METRICS.mkdir(exist_ok=True) # create the metrics directory if it doesn't exist
MODELS.mkdir(exist_ok=True) # create the models directory if it doesn't exist


# ---- CONFIGURATION ----
# Define the configuration for the model
TARGET_RECALL = 0.90
BATCH_SIZE = 8 # the batch size is the number of samples processed before the model is updated
MAX_EPOCHS = 100 # the maximum number of epochs to train the model


# ---- lOAD & PREPROCESS DATA ----
categories = ["COVID-19", "Normal"]
# track all image paths too
paths, x, y = [], [], []
widths, heights = [], []


# compute the mean image size
# (assumes all the images are the same size)
widths, heights = [], []
for c in categories: 
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for img_path in glob.glob(str(DATA_PATH / c / ext)):
            with Image.open(img_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
if not widths:
    raise RuntimeError("No images found in {DATA_PATH}.")

img_width = int(np.mean(widths) /5) # the mean width divided by 5
img_height = int(np.mean(heights) /5) # the mean height divided by 5
# resize to  5% of the mean size
print(f"Resizing images to {img_width}x{img_height} pixels.")

# load the images and Labels into numpy arrays
# x: the images and y: the labels
paths, x, y = [], [], []
for c in categories :
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for img_path in glob.glob(str(DATA_PATH / c / ext)):
            img = Image.open(img_path).convert("L").resize((img_width, img_height))
            x.append(np.array(img))
            y.append(c)
            paths.append(img_path)  # FIX: Add the image path to the paths list
            
x = np.stack(x, axis=0) # stack the images into a numpy array
y = np.array(y) # convert the labels to a numpy array
paths = np.array(paths) # convert the paths to a numpy array
print("All Labels: ", y.shape)# print the shape of the labels
print("All Images: ", x.shape)# print the shape of the images
print("All Paths: ", paths.shape)# print the shape of the paths

# split the data into training and testing sets
x_train, x_test, y_train, y_test, paths_train, paths_test = train_test_split(
    x, y, paths, test_size=0.2, random_state=43, stratify=y
) # stratify ensures that the same proportion of each class is present in both sets
print("Train Labels: ", y_train.shape) # print the shape of the training labels
print("Train Images: ", x_train.shape) # print the shape of the training images
print("Test Labels: ", y_test.shape) # print the shape of the testing labels
print("Test Images: ", x_test.shape) # print the shape of the testing images


# ---- Save DATASET INFO ----
# save the dataset information to a json file
dataset_info = {
   "original_image_shape": list(x.shape[1:]),
   "train_size": len(x_train),
   "test_size": len(x_test),
   "train_class_counts": Counter(y_train.tolist()),
   "test_class_counts": Counter(y_test.tolist()),
   "resize_to": {"width": img_width, "height": img_height},
   "label_encoder_classes": categories, 
}
with open(METRICS / "dataset_info.json", "w") as f:
    json.dump(dataset_info, f, indent=2)
# print the dataset information
print(json.dumps(dataset_info, indent=2))

# Normalise and Reshape the images
x_train = x_train.astype("float32") / 255.0 # normalise the images to the range [0, 1]
x_test = x_test.astype("float32") / 255.0 # normalise the images to the range [0, 1]
x_train = x_train.reshape(-1, img_width, img_height, 1) # reshape the images to the shape (batch_size, width, height, channels)
x_test = x_test.reshape(-1, img_width, img_height, 1) # reshape the images to the shape (batch_size, width, height, channels)
print("Reshaped Train Images: ", x_train.shape) # print the shape of the training images

# --- Save Normalization STATISTICS ---
# save the normalization statistics to a json file
normalization_stats = {
    "mean": float(np.mean(x_train)),
    "std": float(np.std(x_train)),
    "scale_divisor": 255.0,
    "train_pixel_mean": float(np.mean(x_train)),
    "train_pixel_std": float(np.std(x_train)),
    "test_pixel_mean": float(np.mean(x_test)),
}
with open(METRICS / "normalization_stats.json", "w") as f:
    json.dump(normalization_stats, f, indent=2)
    
# Encode the Labels
# encode the labels to integers
label_encoder = LabelEncoder().fit(y_train) # fit the label encoder to the training labels
y_train = label_encoder.transform(y_train) # transform the training labels to integers
y_test = label_encoder.transform(y_test) # transform the testing labels to integers

# Validation Carve-out
# split the training data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=43, stratify=y_train
) # stratify ensures that the same proportion of each class is present in both sets


# ---Model ARCHITECTURE---
# Define the model architecture
# the model is a sequential model with 3 convolutional layers, 3 max pooling layers, and 3 desnse layers
def create_model():
    model = Sequential([
        layers.Input((img_width, img_height, 1)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ])
    # compile the model
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    return model

model = create_model() # create the model
model.summary() # print the model summary

# ---- DATA AUGMENTATION ----
# Define the data augmentation parameters
data_gen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)
data_gen.fit(x_train) # fit the data generator to the training data

# -- Save DATA AUGMENTATION INFO --
# save the data augmentation information to a json file
data_augmentation_info = {
    "rotation_range": 10,
    "width_shift_range": 0.05,
    "height_shift_range": 0.05,
    "shear_range": 0.2,
    "zoom_range": 0.2,
    "horizontal_flip": True,
}
with open(METRICS / "data_augmentation_info.json", "w") as f:
    json.dump(data_augmentation_info, f, indent=2)

callbacks = [
    EarlyStopping(
        monitor='val_accuracy', 
        patience=5,
        restore_best_weights=True,
        verbose=1 # restore the best weights after training
    ),
    ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=3, 
        min_lr=1e-6,
        verbose=1
    ),
]


# ---- TRAINING ----
# Train the model
history = model.fit(
    data_gen.flow(x_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(x_val, y_val),
    epochs=MAX_EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# Save the model
model.save(MODELS / "covid19_model.keras") # save the model to a keras file


# Save the training history
history_df = pd.DataFrame(history.history)
history_df.to_csv(METRICS / "training_history.csv", index=False) # convert the history to a pandas dataframe

# Threshold selection
y_val_pred = model.predict(x_val).flatten()
fpr, tpr, thresholds = roc_curve(y_val, y_val_pred)
idx = np.where(tpr >= TARGET_RECALL)[0][0]
opt_thresh = float(thresholds[idx])
print(f"Threshold for >= {TARGET_RECALL*100:.0f}% recall: {opt_thresh:.3f}")
with open(METRICS / "optimal_threshold.json", "w") as f:
    json.dump({
        "target_recall": TARGET_RECALL,
        "optimal_threshold": opt_thresh,
        "val_true_positive_rate": float(tpr[idx]),
        "val_false_positive_rate": float(fpr[idx])
    }, f, indent=2)

# Final evaluation
y_test_pred = model.predict(x_test).flatten()
y_test_pred_labels = (y_test_pred >= opt_thresh).astype(int)

# --- Save per‐image test results ---
results_df = pd.DataFrame({
    "filepath":             paths_test,
    "true_label":           label_encoder.inverse_transform(y_test),
    "predicted_prob":       y_test_pred,
    "predicted_label":      label_encoder.inverse_transform(y_test_pred_labels),
})
results_df.to_csv(METRICS/"test_results.csv", index=False)

# Classification report
report = classification_report(y_test, y_test_pred_labels, target_names=categories)
with open(METRICS / "classification_report.txt", "w") as f:
    f.write(report)
print(report)

# Confusion matrix
cm = confusion_matrix(y_test, y_test_pred_labels)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=categories, yticklabels=categories)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(METRICS / "confusion_matrix.png")
plt.close()

# Accuracy & F1 Score
accuracy = accuracy_score(y_test, y_test_pred_labels)
f1 = f1_score(y_test, y_test_pred_labels)
with open(METRICS / "performance_scores.json", "w") as f:
    json.dump({
        "accuracy": accuracy,
        "f1_score": f1
    }, f, indent=2)

# ROC & AUC
auc = roc_auc_score(y_test, y_test_pred)
with open(METRICS / "roc_auc.txt", "w") as f:
    f.write(f"test_auc={auc:.3f}\n")
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0,1], [0,1], "--", color="gray")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.savefig(METRICS / "roc_curve.png")
plt.close()

# Precision–Recall & AP
precision, recall, _ = precision_recall_curve(y_test, y_test_pred)
ap = average_precision_score(y_test, y_test_pred)
plt.figure()
plt.plot(recall, precision, label=f"AP = {ap:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.legend(loc="lower left")
plt.savefig(METRICS / "pr_curve.png")
plt.close()

# Calibration curve
prob_true, prob_pred = calibration_curve(y_test, y_test_pred, n_bins=10)
plt.figure()
plt.plot(prob_pred, prob_true, "s-", label="Model")
plt.plot([0,1],[0,1], "--", color="gray", label="Ideal")
plt.xlabel("Mean Predicted Prob")
plt.ylabel("Fraction Positives")
plt.title("Calibration Curve")
plt.legend()
plt.savefig(METRICS / "calibration_curve.png")
plt.close()

# Threshold scan
ths = np.linspace(0.5, 1.0, 101)
accs = [accuracy_score(y_test, (y_test_pred >= t).astype(int)) for t in ths]
f1s = [f1_score(y_test, (y_test_pred >= t).astype(int)) for t in ths]
df_scan = pd.DataFrame({"threshold": ths, "accuracy": accs, "f1_score": f1s})
df_scan.to_csv(METRICS / "threshold_scan.csv", index=False)

print("All metrics and plots saved to:", METRICS)