import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
import joblib

# Load combined features
df = pd.read_csv("fatigue_detection/classical/features_all.csv")
print(f"Total samples: {len(df)}")
print(f"Sofia:  {len(df[df['person'] == 'sofia'])}")
print(f"Matteo: {len(df[df['person'] == 'matteo'])}")
print(f"Awake:  {len(df[df['label'] == 'awake'])}")
print(f"Sleepy: {len(df[df['label'] == 'sleepy'])}")
print()

feature_cols = [
    "ear", "mar", "head_ratio",
    "ear_rolling_mean", "ear_rolling_std",
    "mar_rolling_mean", "mar_rolling_std",
    "head_ratio_rolling_mean", "head_ratio_rolling_std",
    "ear_velocity", "eye_closed", "blink_rolling_sum", "perclos"
]

X = df[feature_cols].values
y = LabelEncoder().fit_transform(df["label"].values)

# =============================================
# APPROACH 1: Per-video aggregated stats
# =============================================
print("=" * 50)
print("APPROACH 1: Per-video statistics")
print("=" * 50)

video_agg = df.groupby(["video", "person", "label"]).agg(
    ear_mean=("ear", "mean"),
    ear_std=("ear", "std"),
    ear_min=("ear", "min"),
    mar_mean=("mar", "mean"),
    mar_std=("mar", "std"),
    mar_max=("mar", "max"),
    head_mean=("head_ratio", "mean"),
    head_std=("head_ratio", "std"),
    perclos_mean=("perclos", "mean"),
    blink_sum_mean=("blink_rolling_sum", "mean"),
).reset_index()

print(f"Total videos: {len(video_agg)}")

X_vid = video_agg.drop(columns=["video", "person", "label"]).values
y_vid = LabelEncoder().fit_transform(video_agg["label"].values)

scaler_vid = StandardScaler()
X_vid_scaled = scaler_vid.fit_transform(X_vid)

classifiers_vid = {
    "SVM (RBF)": SVC(kernel="rbf", C=1.0, gamma="scale"),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
}

for name, clf in classifiers_vid.items():
    scores = cross_val_score(clf, X_vid_scaled, y_vid, cv=10, scoring="accuracy")
    print(f"{name}: accuracy = {scores.mean():.3f} (+/- {scores.std():.3f})")

print()

# =============================================
# APPROACH 2: Per-frame (split by video)
# =============================================
print("=" * 50)
print("APPROACH 2: Per-frame classification")
print("=" * 50)

# Split by video, stratify by person to ensure both people in train and test
videos_sofia = df[df["person"] == "sofia"]["video"].unique().tolist()
videos_matteo = df[df["person"] == "matteo"]["video"].unique().tolist()

np.random.seed(42)
np.random.shuffle(videos_sofia)
np.random.shuffle(videos_matteo)

# 80/20 split per person
s_split = int(0.8 * len(videos_sofia))
m_split = int(0.8 * len(videos_matteo))

train_videos = list(videos_sofia[:s_split]) + list(videos_matteo[:m_split])
test_videos = list(videos_sofia[s_split:]) + list(videos_matteo[m_split:])

train_mask = df["video"].isin(train_videos)
test_mask = df["video"].isin(test_videos)

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"Train: {len(X_train)} frames from {len(train_videos)} videos")
print(f"Test:  {len(X_test)} frames from {len(test_videos)} videos")
print()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

classifiers = {
    "SVM (RBF)": SVC(kernel="rbf", C=10.0, gamma="scale"),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    "KNN (k=7)": KNeighborsClassifier(n_neighbors=7),
}

best_acc = 0
best_name = ""
best_clf = None

for name, clf in classifiers.items():
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"--- {name} ---")
    print(f"Accuracy: {acc:.3f}")
    print(classification_report(y_test, y_pred, target_names=["awake", "sleepy"]))

    if acc > best_acc:
        best_acc = acc
        best_name = name
        best_clf = clf

# Feature importance
rf = classifiers["Random Forest"]
print("--- Feature Importance (Random Forest) ---")
for feat, imp in sorted(zip(feature_cols, rf.feature_importances_), key=lambda x: -x[1]):
    print(f"  {feat:30s} {imp:.4f}")

# Save best model
output_dir = "fatigue_detection/classical"
joblib.dump(best_clf, f"{output_dir}/best_model_all.pkl")
joblib.dump(scaler, f"{output_dir}/scaler_all.pkl")
print(f"\nBest model: {best_name} ({best_acc:.3f})")
print(f"Saved to {output_dir}/best_model_all.pkl")