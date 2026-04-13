import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

# Load features
df = pd.read_csv("fatigue_detection/classical/features.csv")
print(f"Total samples: {len(df)}")
print(f"Awake: {len(df[df['label'] == 'awake'])}")
print(f"Sleepy: {len(df[df['label'] == 'sleepy'])}")
print()

# --- Per-video aggregated features (more robust) ---
print("=" * 50)
print("APPROACH 1: Per-video statistics")
print("=" * 50)

video_features = df.groupby(["video", "label"]).agg(
    ear_mean=("ear", "mean"),
    ear_std=("ear", "std"),
    ear_min=("ear", "min"),
    mar_mean=("mar", "mean"),
    mar_std=("mar", "std"),
    mar_max=("mar", "max"),
    head_mean=("head_ratio", "mean"),
    head_std=("head_ratio", "std"),
).reset_index()

print(f"Videos: {len(video_features)}")
print()

X_vid = video_features.drop(columns=["video", "label"]).values
y_vid = LabelEncoder().fit_transform(video_features["label"].values)  # awake=0, sleepy=1

# Cross-validation on video level (small dataset so use leave-one-out style)
from sklearn.model_selection import LeaveOneOut, cross_val_predict

classifiers = {
    "SVM (RBF)": SVC(kernel="rbf", C=1.0, gamma="scale"),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
}

scaler = StandardScaler()
X_vid_scaled = scaler.fit_transform(X_vid)

for name, clf in classifiers.items():
    scores = cross_val_score(clf, X_vid_scaled, y_vid, cv=min(10, len(X_vid)), scoring="accuracy")
    print(f"{name}: accuracy = {scores.mean():.3f} (+/- {scores.std():.3f})")

print()

# --- Per-frame approach ---
print("=" * 50)
print("APPROACH 2: Per-frame classification")
print("=" * 50)

X = df[["ear", "mar", "head_ratio"]].values
y = LabelEncoder().fit_transform(df["label"].values)

# Split by VIDEO not by frame (avoid data leakage)
videos = df["video"].unique()
np.random.seed(42)
np.random.shuffle(videos)
split = int(0.8 * len(videos))
train_videos = videos[:split]
test_videos = videos[split:]

train_mask = df["video"].isin(train_videos)
test_mask = df["video"].isin(test_videos)

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"Train: {len(X_train)} frames from {len(train_videos)} videos")
print(f"Test:  {len(X_test)} frames from {len(test_videos)} videos")
print()

scaler_frame = StandardScaler()
X_train_scaled = scaler_frame.fit_transform(X_train)
X_test_scaled = scaler_frame.transform(X_test)

best_acc = 0
best_name = ""

for name, clf in classifiers.items():
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc:.3f}")
    print(classification_report(y_test, y_pred, target_names=["awake", "sleepy"]))

    if acc > best_acc:
        best_acc = acc
        best_name = name
        best_clf = clf

# Save the best model
output_dir = "fatigue_detection/classical"
joblib.dump(best_clf, os.path.join(output_dir, "best_model.pkl"))
joblib.dump(scaler_frame, os.path.join(output_dir, "scaler.pkl"))
print(f"\nBest model: {best_name} ({best_acc:.3f})")
print(f"Saved to {output_dir}/best_model.pkl")