import os
import argparse
import json
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    classification_report,
)
import matplotlib.pyplot as plt

try:
    from xgboost import XGBClassifier
except ImportError as e:
    raise SystemExit(
        "xgboost is required for this script. Install with:\n"
        "  python -m pip install xgboost==1.7.6"
    ) from e


def load_dataset(path: str, label_col: str = "ClinSigSimple") -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    if label_col not in df.columns:
        raise ValueError(f"Expected '{label_col}' in dataset at {path}")

    # Ensure binary labels
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    df = df[df[label_col].isin([0, 1])].copy()

    X = df.drop(columns=[label_col])
    y = df[label_col].astype(int)

    # Drop identifiers
    for ident in ["AlleleID"]:
        if ident in X.columns:
            X = X.drop(columns=[ident])

    # One-hot encode any remaining categoricals
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # Replace inf/nan
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return X, y


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )

    thresh = cm.max() / 2.0 if cm.size else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main(
    input_csv: str,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
) -> None:
    print(f"[xgb_train] Loading dataset: {input_csv}")
    X, y = load_dataset(input_csv)
    print(f"[xgb_train] Shape after encoding: X={X.shape}, y={y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Configure a reasonably strong baseline classifier
    clf = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=0,
        tree_method="hist",  # fast default; uses CPU
    )

    print("[xgb_train] Training XGBClassifier...")
    clf.fit(X_train, y_train)

    print("[xgb_train] Evaluating on test set...")
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = float("nan")

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print(f"[xgb_train] Accuracy: {acc:.4f}")
    print(f"[xgb_train] Precision: {pr:.4f}  Recall: {rc:.4f}  F1: {f1:.4f}  ROC AUC: {auc:.4f}")
    print("[xgb_train] Classification report:")
    print(classification_report(y_test, y_pred, digits=4))

    # Validation via CV on training fold (to avoid test leakage)
    print("[xgb_train] Cross-validation (StratifiedKFold=5) on training split...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = []
    for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X_train, y_train), start=1):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]
        clf_cv = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state + fold_idx,
            n_jobs=0,
            tree_method="hist",
        )
        clf_cv.fit(X_tr, y_tr)
        y_va_pred = (clf_cv.predict_proba(X_va)[:, 1] >= 0.5).astype(int)
        acc_va = accuracy_score(y_va, y_va_pred)
        cv_scores.append(acc_va)
    cv_scores = np.array(cv_scores)
    print(f"[xgb_train] CV Accuracy: mean={cv_scores.mean():.4f}, std={cv_scores.std():.4f}")

    # Outputs
    outputs_dir = "data"
    os.makedirs(outputs_dir, exist_ok=True)
    plot_confusion_matrix(cm, labels=["0", "1"], output_path=os.path.join(outputs_dir, "xgb_confusion_matrix.png"))

    metrics = {
        "accuracy": acc,
        "precision": pr,
        "recall": rc,
        "f1": f1,
        "roc_auc": auc,
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "n_features": X.shape[1],
        "n_rows": X.shape[0],
    }
    with open(os.path.join(outputs_dir, "xgb_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[xgb_train] Wrote metrics to {os.path.join(outputs_dir, 'xgb_metrics.json')}")

    # Feature importances
    try:
        importances = clf.feature_importances_
        top_k = min(30, len(importances))
        idx = np.argsort(importances)[::-1][:top_k]
        feat_imp = pd.DataFrame({"feature": X.columns[idx], "importance": importances[idx]})
        feat_imp.to_csv(os.path.join(outputs_dir, "xgb_feature_importances_top30.csv"), index=False)
        print(f"[xgb_train] Saved top-{top_k} feature importances.")
    except Exception:
        pass

    # Save model
    model_path = os.path.join(outputs_dir, "xgb_model.json")
    clf.save_model(model_path)
    print(f"[xgb_train] Saved model to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost on encoded variant dataset")
    parser.add_argument("--input", default="data/variant_summary_encoded.csv", help="Path to encoded CSV with ClinSigSimple")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split fraction (default 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n_estimators", type=int, default=500, help="Number of trees")
    parser.add_argument("--learning_rate", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--max_depth", type=int, default=6, help="Tree max depth")
    args = parser.parse_args()

    main(
        input_csv=args.input,
        test_size=args.test_size,
        random_state=args.seed,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
    )

