import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed

def calc_wap(df):
    wap = (df["bid_price1"] * df["ask_size1"] + df["ask_price1"] * df["bid_size1"]) / (
        df["bid_size1"] + df["ask_size1"]
    )
    return wap


def calc_wap2(df):
    wap = (df["bid_price2"] * df["ask_size2"] + df["ask_price2"] * df["bid_size2"]) / (
        df["bid_size2"] + df["ask_size2"]
    )
    return wap


def log_return(list_stock_prices):
    return np.log(list_stock_prices).diff()


def realized_volatility(series):
    return np.sqrt(np.sum(series**2))


def count_unique(series):
    return len(np.unique(series))


class LightGBMETL:
    def __init__(self, orderbook_path: Path, train_path: Path):
        self.orderbook_path = Path(orderbook_path)
        self.train_path = Path(train_path)

        self._train = pd.read_csv(self.train_path)
        self.list_stock_ids = sorted(self._train["stock_id"].unique())

    @property
    def train(self) -> pd.DataFrame:
        return self._train

    def _preprocessor_book(self, file_path):
        df = pd.read_parquet(file_path)

        # Enforce strict chronological order and reject duplicate bucket seconds
        if df.duplicated(subset=["time_id", "seconds_in_bucket"]).any():
            raise ValueError(f"Duplicate time_id/seconds_in_bucket entries detected in {file_path}")
        df = df.sort_values(["time_id", "seconds_in_bucket"], kind="mergesort").reset_index(drop=True)

        # calculate return etc
        df["wap"] = calc_wap(df)
        df["log_return"] = df.groupby("time_id")["wap"].transform(log_return)

        df["wap2"] = calc_wap2(df)
        df["log_return2"] = df.groupby("time_id")["wap2"].transform(log_return)

        df["wap_balance"] = abs(df["wap"] - df["wap2"])

        df["price_spread"] = (df["ask_price1"] - df["bid_price1"]) / (
            (df["ask_price1"] + df["bid_price1"]) / 2
        )
        df["bid_spread"] = df["bid_price1"] - df["bid_price2"]
        df["ask_spread"] = df["ask_price1"] - df["ask_price2"]
        df["total_volume"] = (df["ask_size1"] + df["ask_size2"]) + (
            df["bid_size1"] + df["bid_size2"]
        )
        df["volume_imbalance"] = abs(
            (df["ask_size1"] + df["ask_size2"]) - (df["bid_size1"] + df["bid_size2"])
        )

        # dict for aggregate
        create_feature_dict = {
            "log_return": [realized_volatility],
            "log_return2": [realized_volatility],
            "wap_balance": [np.mean],
            "price_spread": [np.mean],
            "bid_spread": [np.mean],
            "ask_spread": [np.mean],
            "volume_imbalance": [np.mean],
            "total_volume": [np.mean],
            "wap": [np.mean],
        }

        #####groupby / all seconds
        df_feature = pd.DataFrame(
            df.groupby(["time_id"]).agg(create_feature_dict)
        ).reset_index()

        df_feature.columns = [
            "_".join(col) for col in df_feature.columns
        ]  # time_id is changed to time_id_

        ######groupby / last XX seconds
        last_seconds = [300]

        for second in last_seconds:
            second = 600 - second

            df_feature_sec = pd.DataFrame(
                df.query(f"seconds_in_bucket >= {second}")
                .groupby(["time_id"])
                .agg(create_feature_dict)
            ).reset_index()

            df_feature_sec.columns = [
                "_".join(col) for col in df_feature_sec.columns
            ]  # time_id is changed to time_id_

            df_feature_sec = df_feature_sec.add_suffix("_" + str(second))

            df_feature = pd.merge(
                df_feature,
                df_feature_sec,
                how="left",
                left_on="time_id_",
                right_on=f"time_id__{second}",
            )
            df_feature = df_feature.drop([f"time_id__{second}"], axis=1)

        # create row_id
        file_str = str(file_path).replace("\\", "/")
        if "stock_id=" in file_str:
            stock_id = file_str.split("stock_id=")[1].split("/")[0]
        elif "=" in file_str:
            stock_id = file_str.split("=")[1].split("/")[0]
        else:
            stock_id = Path(file_path).stem
        df_feature["row_id"] = df_feature["time_id_"].apply(lambda x: f"{stock_id}-{x}")
        df_feature = df_feature.drop(["time_id_"], axis=1)

        return df_feature

    def preprocessor(self, stock_ids=None) -> pd.DataFrame:
        stocks = self.list_stock_ids if stock_ids is None else stock_ids

        def for_joblib(stock_id):
            file_path_book = self.orderbook_path / f"stock_id={stock_id}"
            return self._preprocessor_book(file_path_book.as_posix())

        dfs = Parallel(n_jobs=-1, verbose=0)(
            delayed(for_joblib)(stock_id) for stock_id in stocks
        )

        return pd.concat(dfs, ignore_index=True)
