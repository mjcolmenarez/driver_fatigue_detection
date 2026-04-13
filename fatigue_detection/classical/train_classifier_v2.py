import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load features
df = pd.read_csv("fatigue_detection/classical/features_v2.csv")
print(f"Total samples: {len(df)}")
print(f"Awake: {len(df[df['label'] == 'awake'])}")
print(f"Sleepy: {len(df[df['label'] == 'sleepy'])}")
print()

# Feature columns (exclude metadata)
feature_cols = [
    "ear", "mar", "head_ratio",
    "ear_rolling_mean", "ear_rolling_std",
    "mar_rolling_mean", "mar_rolling_std",
    "head_ratio_rolling_mean", "head_ratio_rolling_std",
    "ear_velocity", "eye_closed", "blink_rolling_sum", "perclos"
]

X = df[feature_cols].values
y = LabelEncoder().fit_transform(df["label"].values)

# Split by VIDEO to avoid data leakage
videos = df["video"].unique().tolist()
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

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

classifiers = {
    "SVM (RBF)": SVC(kernel="rbf", C=10.0, gamma="scale"),
    "SVM (Linear)": SVC(kernel="linear", C=1.0),
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

# Feature importance (if Random Forest)
rf = classifiers["Random Forest"]
print("--- Feature Importance (Random Forest) ---")
for feat, imp in sorted(zip(feature_cols, rf.feature_importances_), key=lambda x: -x[1]):
    print(f"  {feat:30s} {imp:.4f}")

# Save best model
output_dir = "fatigue_detection/classical"
joblib.dump(best_clf, f"{output_dir}/best_model_v2.pkl")
joblib.dump(scaler, f"{output_dir}/scaler_v2.pkl")
print(f"\nBest model: {best_name} ({best_acc:.3f})")
print(f"Saved to {output_dir}/best_model_v2.pkl")