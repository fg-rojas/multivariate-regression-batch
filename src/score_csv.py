#!/usr/bin/env python3
# score_csv.py — aplica model.pkl a un CSV y guarda en target_pred_HHMMSS.csv

import argparse
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

def load_model(path: Path):
    return joblib.load(path)


def load_manifest_features(manifest_path: Path | None):
    if not manifest_path:
        return None
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        feats = manifest.get("features")
        if isinstance(feats, list) and all(isinstance(x, str) for x in feats):
            return feats
    except Exception:
        pass
    return None


def predict(model_obj, X: np.ndarray) -> np.ndarray:
    if isinstance(model_obj, dict) and "model" in model_obj:
        kind = model_obj["model"]
        if kind == "Ensemble":
            w = float(model_obj["w"])
            return w * model_obj["xgb"].predict(X) + (1.0 - w) * model_obj["hgbr"].predict(X)
        return model_obj["estimator"].predict(X)
    return model_obj.predict(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="ruta a model.pkl")
    ap.add_argument("--input_csv", required=True, help="CSV con features")
    ap.add_argument("--output_csv", default="out/target_pred.csv", help=f"CSV de salida (se añadirá sufijo _YYYYmmdd_HHMMSS)")
    ap.add_argument("--manifest", default="model/manifest.json", help="ruta a model/manifest.json")
    args = ap.parse_args()

    model = load_model(Path(args.model))
    df = pd.read_csv(args.input_csv)

    features = load_manifest_features(Path(args.manifest)) if args.manifest else None

    if not features:
        raise ValueError(
            "No se pudo determinar la lista de features. "
            "Para solucionarlo, pasa --manifest con 'features'."
        )

    # Validación: CSV con todas las columnas requeridas
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas para hacer predicciones del modelo en el CSV: {missing}")

    # Construir X en el orden del manifest/model
    X = df.loc[:, features].to_numpy(dtype=np.float32, copy=False)
    yhat = predict(model, X)

    # Timestamp de bogotá para nombrar archivo de salida
    ts = datetime.now(ZoneInfo("America/Bogota")).strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output_csv)

    # si no trae extensión, por defecto .csv
    suffix = out_path.suffix if out_path.suffix else ".csv"
    out_with_ts = out_path.with_name(f"{out_path.stem}_{ts}{suffix}")

    # Guardar
    pd.DataFrame({"target_pred": yhat}).to_csv(out_with_ts, index=False)
    print(f"[SUCCESS]: {len(df)} filas insertadas en: {out_with_ts} a las {ts}")

if __name__ == "__main__":
    main()
