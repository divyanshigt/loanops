import json

import pytest
import pandas as pd
import scripts.simulate_drift as simulator
from src.retrain import prepare_drift_training
from src.validate_data import validate, validate_batch


def batch_frame():
    return pd.DataFrame({
        "Loan_ID": ["LP1", "LP2"],
        "Gender": [None, "Male"],
        "Married": ["Yes", "No"],
        "Dependents": ["0", "1"],
        "Education": ["Graduate", "Not Graduate"],
        "Self_Employed": ["No", "No"],
        "ApplicantIncome": [5000.0, 4000.0],
        "CoapplicantIncome": [1000.0, 0.0],
        "LoanAmount": [100.0, 80.0],
        "Loan_Amount_Term": [360.0, 360.0],
        "Credit_History": [1.0, 0.0],
        "Property_Area": ["Urban", "Rural"],
        "Loan_Status": [1, 0],
    })


def test_schema():
    assert len(validate()) > 0


def test_batch_validation_reports_quality_summary(tmp_path):
    source = tmp_path / "batch.csv"
    report_path = tmp_path / "batch_report.json"
    batch_frame().to_csv(source, index=False)

    report = validate_batch(source, report_path)

    assert report["valid"] is True
    assert report["dataset_size"] == {"rows": 2, "columns": 13}
    assert report["missing_values"]["Gender"]["count"] == 1
    assert report["duplicate_ids"] == 0
    assert report["target_distribution"]["1"]["count"] == 1
    assert report["target_distribution"]["0"]["count"] == 1
    assert json.loads(report_path.read_text(encoding="utf-8"))["valid"] is True


def test_batch_validation_rejects_duplicate_ids_and_records(tmp_path):
    source = tmp_path / "duplicate_batch.csv"
    report_path = tmp_path / "duplicate_report.json"
    batch = pd.concat([batch_frame().iloc[[0]], batch_frame().iloc[[0]]])
    batch.to_csv(source, index=False)

    with pytest.raises(ValueError, match="Batch validation failed"):
        validate_batch(source, report_path)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["valid"] is False
    assert report["duplicate_records"] == 1
    assert report["duplicate_ids"] == 1


def test_drift_simulation_preserves_production_history(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    test_frame = pd.DataFrame({
        "Loan_ID": ["LP1"],
        "Gender": ["Male"],
        "Married": ["Yes"],
        "Dependents": ["0"],
        "Education": ["Graduate"],
        "Self_Employed": ["No"],
        "ApplicantIncome": [5000.0],
        "CoapplicantIncome": [1000.0],
        "LoanAmount": [100.0],
        "Loan_Amount_Term": [360.0],
        "Credit_History": [1.0],
        "Property_Area": ["Urban"],
        "Loan_Status": [1],
    })
    test_frame.to_csv(data_dir / "test.csv", index=False)
    production = data_dir / "production_log.csv"
    production.write_text("timestamp,prediction,probability\n2026-01-01,Approved,0.8\n",
                          encoding="utf-8")

    monkeypatch.setattr(simulator, "ROOT", tmp_path)
    simulator.main()

    drift = pd.read_csv(data_dir / "drift_log.csv")
    assert "prediction" not in drift.columns
    assert "probability" not in drift.columns
    assert "Loan_Status" in drift.columns
    assert production.read_text(encoding="utf-8").strip().endswith("Approved,0.8")


def test_retraining_uses_verified_drift_labels():
    base = pd.DataFrame({
        "Gender": ["Male"],
        "Married": ["Yes"],
        "Dependents": ["0"],
        "Education": ["Graduate"],
        "Self_Employed": ["No"],
        "ApplicantIncome": [5000],
        "CoapplicantIncome": [1000],
        "LoanAmount": [100],
        "Loan_Amount_Term": [360],
        "Credit_History": [1],
        "Property_Area": ["Urban"],
        "Loan_Status": [1],
    })
    drift = base.drop(columns=["Loan_Status"]).assign(Loan_Status="N")

    combined = prepare_drift_training(base, drift, drift_weight=1.0)

    assert len(combined) == 2
    assert combined["Loan_Status"].tolist() == [1, 0]
