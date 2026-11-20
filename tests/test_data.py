from src.validate_data import validate
import pandas as pd
import scripts.simulate_drift as simulator

def test_schema():
    assert len(validate()) > 0


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
    assert production.read_text(encoding="utf-8").strip().endswith("Approved,0.8")
