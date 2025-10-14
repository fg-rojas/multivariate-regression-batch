# test to validate the scorer is working correctly

import json
import sys
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_manifest(path: Path, features: list[str]):
    manifest_mock_data = {
        "manifest_version": "1.0",
        "model_version": "test",
        "model_type": "XGB",
        "features": features,
        "metrics": {"oof_rmse": 0.0, "cv_scheme": "n/a"},
        "training": {"created_utc": "2025-10-14T00:00:00Z", "seed": 777},
    }
    path.write_text(json.dumps(manifest_mock_data, indent=2), encoding="utf-8")


def _run_scorer(model_pkl: Path, manifest_json: Path, input_csv: Path, output_csv: Path):
    """Ejecuta el scorer como CLI desde la raíz del repo."""
    cmd = [
        sys.executable,
        "src/score_csv.py",
        "--model", str(model_pkl),
        "--manifest", str(manifest_json),
        "--input_csv", str(input_csv),
        "--output_csv", str(output_csv),
    ]
    return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)


def test_missing_feature_returns_error(tmp_path: Path):
    # it validates that the scorer is working correctly when the input csv has missing features
    feats = ["feature_0", "feature_1", "feature_2"]
    df = pd.DataFrame({"feature_0": [1, 2], "feature_1": [3, 4]})
    data_csv = tmp_path / "in.csv"
    df.to_csv(data_csv, index=False)

    # Dummy model with the expected features
    est = DummyRegressor(strategy="constant", constant=0.0)
    est.fit(pd.DataFrame({"feature_0": [0, 1], "feature_1": [0, 1], "feature_2": [0, 1]}),
            np.array([0, 0]))
    model_pkl = tmp_path / "model.pkl"
    joblib.dump({"features": feats, "estimator": est}, model_pkl)

    # create manifest with the expected features
    manifest_json = tmp_path / "manifest.json"
    _write_manifest(manifest_json, feats)

    # base output directory (the scorer adds timestamp if it writes)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out_base = out_dir / "preds.csv"

    # run scorer
    res = _run_scorer(model_pkl, manifest_json, data_csv, out_base)

    # it should fail due to missing feature
    combined = (res.stdout or "") + (res.stderr or "")
    assert res.returncode != 0, f"Se esperaba error. STDOUT/STDERR:\n{combined}"
    assert "Faltan columnas" in combined, f"Mensaje esperado no encontrado. STDOUT/STDERR:\n{combined}"

    # No debe haberse creado ningún archivo de salida ya que scorer no escribió
    outs = list(out_dir.glob("preds_*.csv"))
    assert len(outs) == 0, f"No deberia existir ningún archivo de salida; encontré: {outs}"
