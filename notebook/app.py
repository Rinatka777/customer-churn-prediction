from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd

# -----------------
# Load saved artifacts
# -----------------
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

app = FastAPI(title="Customer Churn Predictor")

@app.get("/")
def read_root():
    return {"message": "✅ Customer Churn API is running! Visit /docs for Swagger UI."}


# -----------------
# Input Schema with Validation
# -----------------
class CustomerData(BaseModel):
    gender: str = Field(..., pattern="^(Male|Female)$")
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: str = Field(..., pattern="^(Yes|No)$")
    Dependents: str = Field(..., pattern="^(Yes|No)$")
    tenure: float = Field(..., ge=0)
    PhoneService: str = Field(..., pattern="^(Yes|No)$")
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str = Field(..., pattern="^(Yes|No)$")
    PaymentMethod: str
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)

# -----------------
# Prediction Endpoint
# -----------------
@app.post("/predict")
def predict_churn(data: CustomerData):
    # Convert input to DataFrame
    df_input = pd.DataFrame([data.dict()])
    
    # Manual binary mapping
    binary_maps = {
        'YesNo': {'Yes': 1, 'No': 0},
        'Gender': {'Male': 1, 'Female': 0}
    }
    for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
        df_input[col] = df_input[col].map(binary_maps['YesNo'])
    df_input['gender'] = df_input['gender'].map(binary_maps['Gender'])
    
    # One-hot encode multi-class (manual with pandas)
    multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 
                  'OnlineBackup', 'DeviceProtection', 'TechSupport',
                  'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
    
    df_input = pd.get_dummies(df_input, columns=multi_cols, dtype=int)
    
    # Reindex to training columns (missing cols → 0)
    df_input = df_input.reindex(columns=feature_columns, fill_value=0)
    
    # Scale numeric features
    df_input[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(
        df_input[['tenure', 'MonthlyCharges', 'TotalCharges']]
    )
    
    # Predict
    prob = model.predict_proba(df_input)[:, 1][0]
    label = int(prob > 0.5)
    
    return {
        "Churn_Probability": round(prob, 3),
        "Predicted_Label": label
    }
