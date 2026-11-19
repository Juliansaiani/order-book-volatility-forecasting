import numpy as np
import pandas as pd
import pytest

from orderbook_volatility.book_features import LightGBMETL, calc_wap, calc_wap2, realized_volatility
from orderbook_volatility.pipeline import group_partitions, rmspe, fit_models


def fixture_frame():
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({"stock_id": np.repeat([0, 1], 80), "time_id": np.tile(np.arange(80), 2)})
    rv = rng.uniform(.001, .01, len(frame))
    frame["log_return_realized_volatility"] = rv
    frame["price_spread_mean"] = rng.uniform(.0001, .001, len(frame))
    frame["volume_imbalance_mean"] = rng.normal(size=len(frame))
    frame["target"] = 1.1 * rv + .0001
    frame["row_id"] = frame.stock_id.astype(str) + "-" + frame.time_id.astype(str)
    return frame


CONFIG = {"seed": 42, "test_fraction": .2, "validation_fraction": .2, "threads": 1, "lightgbm": {"n_estimators": 5, "num_leaves": 4, "verbosity": -1}}


def test_buckets_never_cross_partitions():
    frame = fixture_frame()
    partitions = group_partitions(frame, CONFIG)
    groups = {k: set(frame.iloc[ix].time_id) for k, ix in partitions.items()}
    assert groups["fit"].isdisjoint(groups["validation"])
    assert groups["fit"].isdisjoint(groups["test"])
    assert groups["validation"].isdisjoint(groups["test"])
    assert sum(map(len, partitions.values())) == len(frame)
    repeated = group_partitions(frame, CONFIG)
    for k in partitions:
        np.testing.assert_array_equal(partitions[k], repeated[k])


def test_final_holdout_labels_cannot_change_model_predictions():
    frame = fixture_frame()
    parts, predict, _, _ = fit_models(frame, CONFIG)
    altered = frame.copy()
    altered.loc[parts["test"], "target"] *= 1000
    _, other_predict, _, _ = fit_models(altered, CONFIG)
    pd.testing.assert_frame_equal(predict(frame.iloc[parts["test"]]), other_predict(frame.iloc[parts["test"]]))


def test_weighted_prices_and_volatility_definition():
    book = pd.DataFrame({"bid_price1": [99.], "ask_price1": [101.], "bid_size1": [3.], "ask_size1": [1.], "bid_price2": [98.], "ask_price2": [102.], "bid_size2": [1.], "ask_size2": [1.]})
    assert calc_wap(book).iloc[0] == 100.5
    assert calc_wap2(book).iloc[0] == 100.
    assert realized_volatility(pd.Series([.03, .04])) == pytest.approx(.05)


def test_metric_is_scale_invariant_and_rejects_invalid_targets():
    assert rmspe([1, 2], [1, 2]) == 0
    assert rmspe([1, 2], [2, 3]) == rmspe([10, 20], [20, 30])
    with pytest.raises(ValueError):
        rmspe([0, 1], [1, 1])


def test_preprocessor_book_sorting_invariance_and_duplicate_rejection(tmp_path):
    # Construct synthetic book updates across two time buckets
    base_rows = []
    for time_id in [100, 200]:
        for sec in range(0, 600, 60):
            base_rows.append({
                "time_id": time_id,
                "seconds_in_bucket": sec,
                "bid_price1": 100.0 + sec * 0.01,
                "ask_price1": 100.05 + sec * 0.01,
                "bid_price2": 99.95 + sec * 0.01,
                "ask_price2": 100.10 + sec * 0.01,
                "bid_size1": 100 + sec,
                "ask_size1": 150 + sec,
                "bid_size2": 200 + sec,
                "ask_size2": 250 + sec,
            })
    df_sorted = pd.DataFrame(base_rows)
    df_shuffled = df_sorted.sample(frac=1.0, random_state=42).reset_index(drop=True)

    sorted_dir = tmp_path / "book" / "stock_id=0"
    sorted_dir.mkdir(parents=True)
    df_sorted.to_parquet(sorted_dir / "part.parquet")

    shuffled_dir = tmp_path / "book" / "stock_id=1"
    shuffled_dir.mkdir(parents=True)
    df_shuffled.to_parquet(shuffled_dir / "part.parquet")

    # Create dummy train.csv
    train_csv = tmp_path / "train.csv"
    pd.DataFrame({"stock_id": [0, 1], "time_id": [100, 200], "target": [0.005, 0.006]}).to_csv(train_csv, index=False)

    etl = LightGBMETL(tmp_path / "book", train_csv)
    res_sorted = etl._preprocessor_book(sorted_dir.as_posix())
    res_shuffled = etl._preprocessor_book(shuffled_dir.as_posix())

    # Replace row_id prefix to compare feature matrices
    res_shuffled["row_id"] = res_shuffled["row_id"].str.replace("1-", "0-")
    pd.testing.assert_frame_equal(res_sorted.sort_values("row_id").reset_index(drop=True),
                                  res_shuffled.sort_values("row_id").reset_index(drop=True))

    # Test duplicate rejection
    dup_dir = tmp_path / "book" / "stock_id=2"
    dup_dir.mkdir(parents=True)
    df_duplicate = pd.concat([df_sorted, df_sorted.iloc[:1]], ignore_index=True)
    df_duplicate.to_parquet(dup_dir / "part.parquet")
    with pytest.raises(ValueError, match="Duplicate time_id/seconds_in_bucket"):
        etl._preprocessor_book(dup_dir.as_posix())


def test_lightgbm_etl_preprocessor_public_interface(tmp_path):
    book_dir = tmp_path / "book_train.parquet" / "stock_id=0"
    book_dir.mkdir(parents=True)
    df_book = pd.DataFrame({
        "time_id": [1, 1],
        "seconds_in_bucket": [0, 50],
        "bid_price1": [10.0, 10.1],
        "ask_price1": [10.2, 10.3],
        "bid_price2": [9.9, 10.0],
        "ask_price2": [10.3, 10.4],
        "bid_size1": [10, 20],
        "ask_size1": [15, 25],
        "bid_size2": [30, 40],
        "ask_size2": [35, 45],
    })
    df_book.to_parquet(book_dir / "part.parquet")

    train_csv = tmp_path / "train.csv"
    pd.DataFrame({"stock_id": [0], "time_id": [1], "target": [0.002]}).to_csv(train_csv, index=False)

    etl = LightGBMETL(tmp_path / "book_train.parquet", train_csv)
    # Call public preprocessor method
    result = etl.preprocessor()
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert "log_return_realized_volatility" in result.columns
    assert result.iloc[0]["row_id"] == "0-1"


def test_weighted_linear_is_invariant_to_predictor_scale():
    frame = fixture_frame()
    parts, predict_base, _, _ = fit_models(frame, CONFIG)
    rescaled = frame.copy()
    rescaled["volume_imbalance_mean"] *= 1e5
    _, predict_scaled, _, _ = fit_models(rescaled, CONFIG)
    p1 = predict_base(frame.iloc[parts["test"]])["weighted_linear"]
    p2 = predict_scaled(rescaled.iloc[parts["test"]])["weighted_linear"]
    np.testing.assert_allclose(p1.to_numpy(), p2.to_numpy(), atol=1e-12, rtol=1e-12)
