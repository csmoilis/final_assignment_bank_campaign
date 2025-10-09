from fastapi import FastAPI
from pydantic import BaseModel
import os
from typing import List, Literal, Optional
import joblib
import numpy as np
import pandas as pd
import requests
import shap
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

# =====================================================
# CONFIG
# =====================================================

# Replace these with your NoCoDB API details
NOCO_API_URL = "https://dun3co-sdc-nocodb.hf.space/api/v2/tables/m39a8axnn3980w9/records"
NOCO_VIEW_ID = "vwjuv5jnaet9npuu"
NOCO_API_TOKEN = os.getenv("NOCODB_TOKEN")

HEADERS = {"xc-token": NOCO_API_TOKEN}

# =====================================================
# MODEL LOADING
# =====================================================

model = joblib.load("model_1mvp.pkl")
app = FastAPI(title="Logistic Regression API 2")

# =====================================================
# DATA SCHEMAS
# =====================================================

class InputData(BaseModel):
    age: int
    balance: float
    day: int
    campaign: int
    job: str
    education: str
    default: Literal["yes", "no", "unknown"]
    housing: Literal["yes", "no", "unknown"]
    loan: Literal["yes", "no", "unknown"]
    months_since_previous_contact: str
    n_previous_contacts: str
    poutcome: str
    had_contact: bool
    is_single: bool
    uknown_contact: bool

class BatchInputData(BaseModel):
    data: List[InputData]

# =====================================================
# HEALTH CHECK
# =====================================================

@app.get("/health")
def health():
    return {"status": "ok"}

# =====================================================
# NOCODB DATA FETCHING
# =====================================================

def fetch_test_data(limit: int = 100):
    """Fetch test or sample data from NoCoDB view."""
    params = {"offset": 0, "limit": limit, "viewId": NOCO_VIEW_ID}
    res = requests.get(NOCO_API_URL, headers=HEADERS, params=params)
    res.raise_for_status()
    data = res.json()["list"]
    return pd.DataFrame(data)

# =====================================================
# PREDICTION ENDPOINT
# =====================================================

@app.post("/predict")
def predict(batch: BatchInputData):
    try:
        X = pd.DataFrame([item.dict() for item in batch.data])
        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1]
        return {
            "predictions": preds.tolist(),
            "probabilities": probs.tolist()
        }
    except Exception as e:
        import traceback
        return {"error": str(e), "trace": traceback.format_exc()}

# =====================================================
# EXPLAINABILITY ENDPOINT
# =====================================================

@app.post("/explain")
def explain(batch: Optional[BatchInputData] = None, limit: int = 100):
    """Generate SHAP values either from provided data or from NoCoDB test data."""
    try:
        if batch:
            X = pd.DataFrame([item.dict() for item in batch.data])
            source = "client batch"
        else:
            X = fetch_test_data(limit=limit)
            source = f"NoCoDB (limit={limit})"

        print(f"[DEBUG] SHAP explain called using {source} | shape={X.shape} | cols={list(X.columns)}")

        # Remove ID and target columns if they exist
        drop_cols = [c for c in ["Id", "y", "target"] if c in X.columns]
        if drop_cols:
            print(f"[DEBUG] Dropping columns not used for prediction: {drop_cols}")
            X = X.drop(columns=drop_cols)

        # Handle pipelines correctly
        if hasattr(model, "named_steps"):
            preprocessor = model.named_steps["preprocessor"]
            classifier = model.named_steps["classifier"]

            X_transformed = preprocessor.transform(X)
            feature_names = preprocessor.get_feature_names_out()

            print(f"[DEBUG] Transformed shape: {X_transformed.shape} | n_features={len(feature_names)}")

            explainer = shap.Explainer(classifier, X_transformed)
            shap_values = explainer(X_transformed)

            shap_summary = pd.DataFrame({
                "feature": feature_names,
                "mean_abs_shap": np.abs(shap_values.values).mean(axis=0)
            }).sort_values("mean_abs_shap", ascending=False)
        else:
            # If model is not a pipeline
            explainer = shap.Explainer(model, X)
            shap_values = explainer(X)
            shap_summary = pd.DataFrame({
                "feature": X.columns,
                "mean_abs_shap": np.abs(shap_values.values).mean(axis=0)
            }).sort_values("mean_abs_shap", ascending=False)

        print(f"[DEBUG] SHAP summary created successfully with {len(shap_summary)} features.")
        return {"n_samples": len(X), "shap_summary": shap_summary.to_dict(orient="records")}

    except Exception as e:
        import traceback
        print("[ERROR] SHAP explain failed:", e)
        print(traceback.format_exc())
        return {"error": str(e), "trace": traceback.format_exc()}


# =====================================================
# METRICS ENDPOINT
# =====================================================

@app.post("/metrics")
def metrics(batch: Optional[BatchInputData] = None, limit: int = 100):
    """
    Compute ROC AUC and threshold analysis using input or NoCoDB test data.
    Assumes the target column 'y' is boolean (True/False).
    """
    # Defaults in case something fails
    roc_auc = None
    pr_auc = None
    thresholds = []
    precision = []
    recall = []

    try:
        # Fetch data from batch or NoCoDB
        if batch:
            X = pd.DataFrame([item.dict() for item in batch.data])
            source = "client batch"
        else:
            X = fetch_test_data(limit=limit)
            source = f"NoCoDB (limit={limit})"

        print(f"[DEBUG] Metrics called using {source} | shape={X.shape}")

        # Ensure target 'y' exists
        if "y" not in X.columns:
            return {"error": "No target column 'y' found in dataset."}

        # Convert boolean target to integer
        y_true = X["y"].astype(int).tolist()
        print(f"[DEBUG] Found {sum(y_true)} positive cases out of {len(y_true)}")
        X = X.drop(columns=["y"])

        # Drop ID if exists
        if "Id" in X.columns:
            X = X.drop(columns=["Id"])

        # Predict probabilities
        y_prob = model.predict_proba(X)[:, 1]

        # Compute metrics
        roc_auc = roc_auc_score(y_true, y_prob)
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)

        print(f"[DEBUG] ROC AUC={roc_auc:.3f} | PR AUC={pr_auc:.3f}")

        return {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "thresholds": thresholds.tolist()[:20],
            "precision": precision.tolist()[:20],
            "recall": recall.tolist()[:20]
        }

    except Exception as e:
        import traceback
        print("[ERROR] Metrics failed:", e)
        print(traceback.format_exc())
        return {"error": str(e), "trace": traceback.format_exc()}


    
@app.get("/coefficients")
def coefficients():
    """
    Return logistic regression coefficients and feature names.
    Works if your model is a pipeline with 'preprocessor' and 'classifier' steps.
    """
    try:
        # Extract classifier and preprocessor
        classifier = model.named_steps["classifier"]
        preprocessor = model.named_steps["preprocessor"]

        # Get feature names after preprocessing
        feature_names = preprocessor.get_feature_names_out()

        # Get coefficients
        coefficients = classifier.coef_[0]

        df = pd.DataFrame({
            "feature": feature_names,
            "coefficient": coefficients.tolist()
        })

        return {"coefficients": df.to_dict(orient="records")}

    except Exception as e:
        import traceback
        return {"error": str(e), "trace": traceback.format_exc()}
