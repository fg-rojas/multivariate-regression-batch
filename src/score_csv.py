#!/usr/bin/env python3
# score_csv.py — aplica model.pkl a un CSV y guarda target_pred.csv


import argparse
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd


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
    ap.add_argument("--output_csv", default="target_pred.csv", help="CSV de salida")
    ap.add_argument("--manifest", default=" model/manifest.json", help="ruta a model/manifest.json")
    args = ap.parse_args()

    model = load_model(Path(args.model))
    df = pd.read_csv(args.input_csv)


    manifest_feats = load_manifest_features(Path(args.manifest)) if args.manifest else None

    # Si no hay manifest.json de modelo de training entonces intenta usar features del modelo .pkl
    model_feats = model["features"]


    # Selección final de la lista de columnas y orden
    features = manifest_feats or model_feats
    if not features:
        raise ValueError(
            "No se pudo determinar la lista de features. "
            "Pasa --manifest con 'features' o guarda 'features' dentro del model.pkl, "
        )

    # Validación: que el CSV tenga todas las columnas requeridas para el modelo
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas para hacer predicciones del modelo en el CSV: {missing}")

    # X como en training y predicción
    X = df.loc[:, features].to_numpy(dtype=np.float32, copy=False)
    yhat = predict(model, X)

    # Guardar
    pd.DataFrame({"target_pred": yhat}).to_csv(args.output_csv, index=False)
    print(f"[SUCCESS]: {len(df)} filas insertadas en: {args.output_csv}")
    print(f"Orden de features utilizado: {features[:10]}{' ...' if len(features) > 10 else ''}")

if __name__ == "__main__":
    main()
