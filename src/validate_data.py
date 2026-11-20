import argparse
import json
import sys
from pathlib import Path

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


def validate_batch(file_path, report_path=None):
    source = Path(file_path)
    if not source.is_absolute():
        source = path(source)
    df = pd.read_csv(source)
    missing_columns = [column for column in REQUIRED if column not in df.columns]
    errors = []
    warnings = []

    if missing_columns:
        errors.append(f"Missing columns: {missing_columns}")
    if df.empty:
        errors.append("Batch must contain at least one row")

    missing_counts = df.isna().sum()
    row_count = len(df)
    missing_values = {
        str(column): {
            "count": int(count),
            "percent": round(float(count / row_count * 100), 2) if row_count else 0.0,
        }
        for column, count in missing_counts.items()
    }
    feature_missing = {
        column: count for column, count in missing_counts.items()
        if count and column not in {"Loan_ID", "Loan_Status"}
    }
    if feature_missing:
        warnings.append(
            "Missing feature values are reported but allowed; the preprocessing "
            "pipeline imputes them."
        )

    duplicate_records = int(df.duplicated().sum())
    if duplicate_records:
        errors.append(f"Found {duplicate_records} duplicate records")

    duplicate_ids = 0
    if "Loan_ID" in df.columns:
        ids = df.loc[df["Loan_ID"].notna(), "Loan_ID"]
        duplicate_ids = int(ids.duplicated().sum())
        if duplicate_ids:
            errors.append(f"Found {duplicate_ids} duplicate Loan_ID values")

    target_distribution = {}
    if "Loan_Status" in df.columns:
        target = df["Loan_Status"]
        counts = target.value_counts(dropna=False)
        target_distribution = {
            "<missing>" if pd.isna(label) else str(label): {
                "count": int(count),
                "percent": round(float(count / row_count * 100), 2) if row_count else 0.0,
            }
            for label, count in counts.items()
        }
        if target.isna().any():
            errors.append("Loan_Status contains missing labels")
        allowed = {"Y", "N", "0", "1", "0.0", "1.0"}
        invalid_labels = sorted({
            str(value).strip().upper()
            for value in target.dropna().unique()
            if str(value).strip().upper() not in allowed
        })
        if invalid_labels:
            errors.append(f"Invalid Loan_Status values: {invalid_labels}")

    for column in ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]:
        if column in df.columns:
            numeric = pd.to_numeric(df[column], errors="coerce")
            invalid_numeric = df[column].notna() & numeric.isna()
            if invalid_numeric.any():
                errors.append(f"Non-numeric values in {column}")
            if (numeric < 0).any():
                errors.append(f"Negative values in {column}")

    report = {
        "file": str(source),
        "valid": not errors,
        "dataset_size": {"rows": row_count, "columns": len(df.columns)},
        "missing_values": missing_values,
        "duplicate_records": duplicate_records,
        "duplicate_ids": duplicate_ids,
        "target_distribution": target_distribution,
        "warnings": warnings,
        "errors": errors,
    }
    destination = Path(report_path) if report_path else path("reports/new_data_validation.json")
    if not destination.is_absolute():
        destination = path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise ValueError(f"Batch validation failed; see {destination}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch",
        help="Validate a new labeled batch and save its data-quality report",
    )
    parser.add_argument(
        "--report",
        help="Optional JSON report path (requires --batch)",
    )
    args = parser.parse_args()
    if args.report and not args.batch:
        parser.error("--report can only be used with --batch")
    try:
        if args.batch:
            validate_batch(args.batch, args.report)
        else:
            validate()
    except Exception as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        raise SystemExit(1)
