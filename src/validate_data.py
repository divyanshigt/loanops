import sys
import pandas as pd
from src.utils import load_params, path

REQUIRED = ["Loan_ID", "Gender", "Married", "Dependents", "Education", "Self_Employed",
            "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
            "Credit_History", "Property_Area", "Loan_Status"]

def validate(file_path=None):
    params = load_params()
    file_path = file_path or path(params["paths"]["data"])
    df = pd.read_csv(file_path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if len(df) < 20:
        raise ValueError("Dataset must contain at least 20 rows")
    for col in ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]:
        if (pd.to_numeric(df[col], errors="coerce") < 0).any():
            raise ValueError(f"Negative values in {col}")
    if not set(df["Loan_Status"].dropna()).issubset({"Y", "N"}):
        raise ValueError("Loan_Status must contain only Y or N")
    print(f"VALID: {file_path} ({len(df)} rows)")
    return df

if __name__ == "__main__":
    try:
        validate()
    except Exception as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        raise SystemExit(1)
