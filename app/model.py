import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_model(data):
    # 🔹 Clean column names
    data.columns = data.columns.str.strip().str.lower().str.replace(" ", "_")

    # 🔹 Auto-detect target column
    label_candidates = [c for c in data.columns if "heart" in c or "target" in c or "risk" in c]
    label_col = label_candidates[0] if label_candidates else data.columns[-1]

    X = data.drop(label_col, axis=1)
    y = data[label_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)

    return model, scaler, acc, report, X_test, y_test
