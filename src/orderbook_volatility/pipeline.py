"""Download, build features, evaluate frozen models, and export a compact report."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .book_features import LightGBMETL


def rmspe(actual, predicted) -> float:
    y, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if y.size == 0 or y.shape != p.shape or not np.all(np.isfinite(y)) or np.any(y <= 0):
        raise ValueError("RMSPE requires matching, nonempty arrays and finite positive targets.")
    if not np.all(np.isfinite(p)):
        raise ValueError("Predictions must be finite.")
    return float(np.sqrt(np.mean(((y - p) / y) ** 2)))


def group_partitions(frame: pd.DataFrame, config: dict) -> dict[str, np.ndarray]:
    """Disjoint anonymized time buckets across fit, validation and final holdout."""
    outer = GroupShuffleSplit(n_splits=1, test_size=config["test_fraction"], random_state=config["seed"])
    development, test = next(outer.split(frame, groups=frame.time_id))
    inner = GroupShuffleSplit(n_splits=1, test_size=config["validation_fraction"], random_state=config["seed"] + 1)
    fit_local, valid_local = next(inner.split(frame.iloc[development], groups=frame.iloc[development].time_id))
    return {"fit": development[fit_local], "validation": development[valid_local], "test": test}


def _fetch(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "orderbook-volatility-research/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def download_data(root: Path, config: dict) -> dict:
    """Pinned source revision, local cache, and a SHA-256 manifest. Raw files stay local."""
    repo, revision = config["source_repo"], config["source_revision"]
    data_root = root / "data"
    prefix = "optiver-realized-volatility-prediction"
    paths = [f"{prefix}/train.csv"]
    for stock in config["stock_ids"]:
        folder = f"{prefix}/book_train.parquet/stock_id={stock}"
        cache = data_root / "book_train.parquet" / f"stock_id={stock}"
        existing = list(cache.glob("*.parquet"))
        if existing:
            paths.append(f"{folder}/{existing[0].name}")
            continue
        api = f"https://api.github.com/repos/{repo}/contents/{folder}?ref={revision}"
        with urllib.request.urlopen(urllib.request.Request(api, headers={"User-Agent": "orderbook-volatility-research/0.1"}), timeout=30) as response:
            entries = json.load(response)
        files = [entry["path"] for entry in entries if entry["name"].endswith(".parquet")]
        if len(files) != 1:
            raise ValueError(f"Expected one order-book partition for stock {stock}.")
        paths.extend(files)
    records = []
    previous_path = data_root / "manifest.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else None
    if previous and previous.get("revision") != revision:
        raise ValueError("Cached source revision differs; use a separate data directory.")
    for remote in paths:
        relative = remote.removeprefix(prefix + "/")
        destination = data_root / relative
        url = f"https://raw.githubusercontent.com/{repo}/{revision}/{urllib.parse.quote(remote)}"
        if not destination.exists():
            print(f"Downloading {relative}", flush=True)
            _fetch(url, destination)
        with destination.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        old = next((f for f in previous.get("files", []) if f["path"] == relative), None) if previous else None
        if old and old["sha256"] != digest:
            raise ValueError(f"Cached file checksum changed: {relative}")
        records.append({"path": relative, "bytes": destination.stat().st_size, "sha256": digest, "url": url})
    manifest = {"dataset": "Optiver Realized Volatility Prediction", "source_repository": repo, "revision": revision, "files": records}
    previous_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_features(root: Path, config: dict) -> pd.DataFrame:
    data = root / "data"
    extractor = LightGBMETL(data / "book_train.parquet", data / "train.csv")
    frames = []
    for stock in config["stock_ids"]:
        print(f"Features: stock {stock}", flush=True)
        frame = extractor._preprocessor_book((data / "book_train.parquet" / f"stock_id={stock}").as_posix())
        frame["stock_id"] = stock
        frame["time_id"] = frame.row_id.str.split("-").str[-1].astype(int)
        frames.append(frame)
    features = pd.concat(frames, ignore_index=True)
    labels = extractor.train[["stock_id", "time_id", "target"]]
    features = features.merge(labels, on=["stock_id", "time_id"], validate="one_to_one")
    features = features.sort_values(["stock_id", "time_id"]).reset_index(drop=True)
    if features[["stock_id", "time_id"]].duplicated().any() or features.isna().any().any():
        raise ValueError("Feature keys must be unique, with no missing feature values.")
    if len(features) == 0 or not np.isfinite(features.select_dtypes("number")).all().all():
        raise ValueError("Features must be nonempty and finite.")
    (root / "artifacts").mkdir(exist_ok=True)
    features.to_parquet(root / "artifacts/features.parquet", index=False)
    return features


def fit_models(features: pd.DataFrame, config: dict):
    """Fit without final-holdout labels; no label-based stock encoding or tuning."""
    parts = group_partitions(features, config)
    fit = features.iloc[parts["fit"]]
    predictors = [c for c in features.columns if c not in {"row_id", "time_id", "target", "stock_id"}]
    wls_columns = ["log_return_realized_volatility", "price_spread_mean", "volume_imbalance_mean"]
    weights = 1 / np.square(fit.target.to_numpy())
    weights /= weights.mean()
    linear = {}
    for stock in sorted(features.stock_id.unique()):
        subset = fit.stock_id == stock
        # Standardizing predictors eliminates the ~1e7 condition-number spread across RV, spread, and volume imbalance.
        linear[int(stock)] = make_pipeline(StandardScaler(), LinearRegression()).fit(
            fit.loc[subset, wls_columns],
            fit.loc[subset, "target"],
            linearregression__sample_weight=weights[subset],
        )
    booster = LGBMRegressor(**config["lightgbm"], random_state=config["seed"], n_jobs=config["threads"], deterministic=True, force_col_wise=True)
    booster.fit(fit[predictors], fit.target, sample_weight=weights)

    def predict(frame):
        result = pd.DataFrame(index=frame.index)
        result["persistence"] = frame["log_return_realized_volatility"]
        result["weighted_linear"] = np.nan
        for stock, model in linear.items():
            subset = frame.stock_id == stock
            if subset.any():
                result.loc[subset, "weighted_linear"] = model.predict(frame.loc[subset, wls_columns])
        result["lightgbm"] = booster.predict(frame[predictors])
        # Volatility is nonnegative. Apply the same fixed floor to all predictions.
        return result.clip(lower=1e-12)

    return parts, predict, booster, predictors


def evaluate(features, config):
    parts, predict, booster, predictors = fit_models(features, config)
    rows, prediction_frames = [], []
    for split in ("validation", "test"):
        frame = features.iloc[parts[split]]
        predicted = predict(frame)
        for model in predicted.columns:
            rows.append({"split": split, "model": model, "rmspe": rmspe(frame.target, predicted[model]), "rows": len(frame)})
        prediction_frames.append(pd.concat([frame[["stock_id", "time_id", "target"]], predicted], axis=1).assign(split=split))
    predictions = pd.concat(prediction_frames).reset_index(drop=True)
    scores = pd.DataFrame(rows)
    per_stock = []
    test = predictions[predictions.split == "test"]
    for stock, sample in test.groupby("stock_id"):
        for model in ("persistence", "weighted_linear", "lightgbm"):
            per_stock.append({"stock_id": int(stock), "model": model, "rmspe": rmspe(sample.target, sample[model]), "rows": len(sample)})
    importance = pd.DataFrame({"feature": predictors, "gain": booster.booster_.feature_importance(importance_type="gain")}).sort_values("gain", ascending=False)
    return scores, predictions, pd.DataFrame(per_stock), importance, parts


def plot_results(scores, predictions, importance, output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    test = predictions[predictions.split == "test"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    s = scores[scores.split == "test"]
    axes[0].barh(s.model, s.rmspe, color=["#94a3b8", "#2563eb", "#0f766e"])
    axes[0].invert_yaxis()
    axes[0].set(xlabel="RMSPE (lower is better)", title="Held-out time buckets")
    for i, value in enumerate(s.rmspe):
        axes[0].text(value, i, f"  {value:.4f}", va="center")
    sample = test.sample(min(1500, len(test)), random_state=42)
    axes[1].scatter(sample.target, sample.lightgbm, s=9, alpha=.4, color="#0f766e", rasterized=True)
    lo, hi = float(sample.target.min()), float(max(sample.target.max(), sample.lightgbm.max()))
    axes[1].plot([lo, hi], [lo, hi], color="#475569", linewidth=1, linestyle="--")
    axes[1].set(xscale="log", yscale="log", xlabel="Observed volatility", ylabel="LightGBM prediction", title="Predicted versus observed")
    fig.suptitle("Order-book volatility forecasting", fontsize=15, fontweight="bold")
    fig.savefig(output / "overview.png", bbox_inches="tight")
    plt.close(fig)
    top = importance.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    ax.barh(top.feature, top.gain, color="#0f766e")
    ax.set(xlabel="Training gain (not causal importance)", title="LightGBM feature importance")
    fig.savefig(output / "feature_importance.png", bbox_inches="tight")
    plt.close(fig)


def bootstrap_differences(predictions, seed=42, repetitions=1000):
    """Paired bucket bootstrap: descriptive uncertainty, not a chronological guarantee."""
    test = predictions[predictions.split == "test"].copy()
    models = ["persistence", "weighted_linear", "lightgbm"]
    for model in models:
        test[model + "_error"] = ((test.target - test[model]) / test.target) ** 2
    errors = test.groupby("time_id")[[m + "_error" for m in models]].sum().to_numpy()
    counts = test.groupby("time_id").size().to_numpy()
    rng = np.random.default_rng(seed)
    samples = np.empty((repetitions, len(models)))
    for i in range(repetitions):
        ix = rng.integers(0, len(counts), size=len(counts))
        samples[i] = np.sqrt(errors[ix].sum(axis=0) / counts[ix].sum())
    rows = []
    for i, model in enumerate(models[1:], 1):
        delta = samples[:, i] - samples[:, 0]
        rows.append({"model": model, "reference": "persistence", "rmspe_difference": rmspe(test.target, test[model]) - rmspe(test.target, test.persistence), "lower_95": float(np.quantile(delta, .025)), "upper_95": float(np.quantile(delta, .975)), "bootstrap_repetitions": repetitions})
    return pd.DataFrame(rows)


def run(root: Path, config_path: Path | None = None) -> dict:
    root = root.resolve()
    config_path = config_path or root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = download_data(root, config)
    features = build_features(root, config)
    scores, predictions, per_stock, importance, parts = evaluate(features, config)
    output = root / "artifacts"
    scores.to_csv(output / "metrics.csv", index=False)
    per_stock.to_csv(output / "per_stock.csv", index=False)
    importance.to_csv(output / "feature_importance.csv", index=False)
    predictions.to_parquet(output / "predictions.parquet", index=False)
    bootstrap_differences(predictions, seed=config["seed"]).to_csv(output / "uncertainty.csv", index=False)
    # Reproducible bucket assignment, with no raw price data in the public repository.
    assignment = pd.concat([features.iloc[ix][["time_id"]].drop_duplicates().assign(split=name) for name, ix in parts.items()])
    assignment.sort_values("time_id").to_csv(output / "bucket_assignment.csv", index=False)
    metadata = {"config": config, "python": platform.python_version(), "packages": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scikit-learn", "lightgbm", "pyarrow"]}, "observations": len(features), "stocks": int(features.stock_id.nunique()), "time_buckets": int(features.time_id.nunique()), "split_rows": {name: len(ix) for name, ix in parts.items()}, "validation": "seeded grouped holdout; anonymized time_id is not a chronological timestamp", "data": manifest}
    (output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    plot_results(scores, predictions, importance, output)
    print(scores.to_string(index=False), flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    run(args.root, args.config)


if __name__ == "__main__":
    main()
