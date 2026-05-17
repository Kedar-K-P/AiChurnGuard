"""
train.py  -  XGBoost churn model training pipeline
FIX: removed deprecated use_label_encoder=False (XGBoost 2.x)
"""
from __future__ import annotations
import argparse, json, logging
from pathlib import Path
import joblib, numpy as np
from sklearn.metrics import (classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier
from data_processing import ALL_FEATURES, TARGET, build_preprocessor, get_feature_names_out, load_or_generate_data

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
MODEL_PATH        = ARTIFACTS_DIR / "xgb_model.joblib"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"
METRICS_PATH      = ARTIFACTS_DIR / "metrics.json"
FEATURE_IMPORTANCE_PATH = ARTIFACTS_DIR / "feature_importance.json"

def train(csv_path: str | None = None) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_or_generate_data(csv_path)
    X, y = df[ALL_FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    logger.info("Train: %d | Test: %d | Churn rate: %.1f%%", len(X_train), len(X_test), y_train.mean()*100)

    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t  = preprocessor.transform(X_test)
    feature_names = get_feature_names_out(preprocessor)

    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / pos

    model = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss", random_state=42,
        n_jobs=-1, tree_method="hist",
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(model, X_train_t, y_train, cv=cv, scoring="roc_auc")
    logger.info("CV ROC-AUC: %.4f +/- %.4f", cv_auc.mean(), cv_auc.std())

    model.fit(X_train_t, y_train, eval_set=[(X_test_t, y_test)], verbose=False)

    y_pred  = model.predict(X_test_t)
    y_proba = model.predict_proba(X_test_t)[:, 1]
    roc_auc   = roc_auc_score(y_test, y_proba)
    f1        = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    logger.info("ROC-AUC: %.4f | F1: %.4f | Precision: %.4f | Recall: %.4f", roc_auc, f1, precision, recall)
    logger.info("\n%s", classification_report(y_test, y_pred, target_names=["Retained","Churned"]))

    metrics = {
        "roc_auc": round(roc_auc,4), "f1_score": round(f1,4),
        "precision": round(precision,4), "recall": round(recall,4),
        "cv_auc_mean": round(float(cv_auc.mean()),4), "cv_auc_std": round(float(cv_auc.std()),4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "train_samples": int(len(X_train)), "test_samples": int(len(X_test)),
        "churn_rate_pct": round(float(y.mean()*100),2),
    }

    importances = model.feature_importances_
    fi_pairs = sorted(zip(feature_names, importances.tolist()), key=lambda x: x[1], reverse=True)
    feature_importance = [{"feature": n, "importance": round(s,6)} for n, s in fi_pairs]

    joblib.dump(model, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    with open(METRICS_PATH, "w") as f: json.dump(metrics, f, indent=2)
    with open(FEATURE_IMPORTANCE_PATH, "w") as f: json.dump(feature_importance, f, indent=2)
    logger.info("Artifacts saved to %s", ARTIFACTS_DIR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None)
    args = parser.parse_args()
    train(csv_path=args.data)
