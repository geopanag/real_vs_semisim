from __future__ import annotations

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

RETAIL_URL = "https://storage.yandexcloud.net/datasouls-ods/materials/9c6913e5/retailhero-uplift.zip"


def prepare_retail_csv(root: str | Path, age_filter: int = 16) -> Path:
    """
    Reproduce the old RetailHero tabular preprocessing while removing all graph outputs.
    Writes:
      <root>/processed/retail.csv
    """
    root = Path(root)
    raw_dir = root / "raw"
    processed_dir = root / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    retail_csv = processed_dir / "retail.csv"
    if retail_csv.is_file():
        return retail_csv

    uplift_train = raw_dir / "data" / "uplift_train.csv"
    clients_csv = raw_dir / "data" / "clients.csv"
    purchases_csv = raw_dir / "data" / "purchases.csv"

    if not (uplift_train.is_file() and clients_csv.is_file() and purchases_csv.is_file()):
        local_zip = raw_dir / "retailhero-uplift.zip"
        if not local_zip.is_file():
            urlretrieve(RETAIL_URL, local_zip)
        with zipfile.ZipFile(local_zip, "r") as zf:
            zf.extractall(raw_dir)

    encoder = OneHotEncoder()

    train = pd.read_csv(uplift_train).set_index("client_id")

    df_features = pd.read_csv(clients_csv)
    df_features["first_redeem_date"] = pd.to_datetime(df_features["first_redeem_date"])
    df_features["first_issue_abs_time"] = (
        pd.to_datetime(df_features["first_issue_date"]) - pd.Timestamp("1970-01-01")
    ) // pd.Timedelta("1s")
    df_features["first_redeem_abs_time"] = (
        pd.to_datetime(df_features["first_redeem_date"]) - pd.Timestamp("1970-01-01")
    ) // pd.Timedelta("1s")
    df_features["redeem_delay"] = (
        df_features["first_redeem_abs_time"] - df_features["first_issue_abs_time"]
    )
    df_features = df_features[df_features["age"] > age_filter]
    df_features = df_features[df_features["redeem_delay"] > 0].reset_index(drop=True)

    one_hot_encoded = encoder.fit_transform(df_features[["gender"]]).toarray()
    encoded_categories = encoder.categories_
    df_encoded = pd.DataFrame(one_hot_encoded, columns=encoded_categories[0])
    df_features = df_features.drop("gender", axis=1)
    columns = list(df_features.columns) + list(encoded_categories[0])
    df_features = pd.concat([df_features, df_encoded], axis=1, ignore_index=True)
    df_features.columns = columns
    df_features = train.join(df_features.set_index("client_id"))
    df_features = df_features[~df_features.age.isna()]

    purchases = pd.read_csv(purchases_csv)
    purchases = purchases[
        [
            "client_id",
            "transaction_id",
            "transaction_datetime",
            "purchase_sum",
            "store_id",
            "product_id",
            "product_quantity",
        ]
    ]
    purchases["transaction_datetime"] = pd.to_datetime(purchases["transaction_datetime"])

    first_redeem_map = dict(zip(df_features.index, df_features["first_redeem_date"]))
    purchases["first_redeem_date"] = purchases["client_id"].map(first_redeem_map)
    purchases = purchases[~purchases["first_redeem_date"].isna()]

    purchases_before = purchases[purchases["transaction_datetime"] < purchases["first_redeem_date"]]
    purchases_after = purchases[purchases["transaction_datetime"] >= purchases["first_redeem_date"]]

    features_before = purchases_before.groupby("transaction_id").agg(
        {"client_id": "first", "purchase_sum": "first", "transaction_datetime": "first"}
    ).reset_index()
    features_before.columns = ["transaction_id", "client_id", "purchase_sum", "transaction_datetime"]
    features_before = features_before.groupby("client_id").agg(
        {"purchase_sum": "mean", "transaction_id": "count", "transaction_datetime": ["max", "min"]}
    )
    features_before.columns = [
        "avg_money_before",
        "total_count_before",
        "last_purchase_before",
        "first_purchase_before",
    ]
    features_before["avg_count_before"] = features_before["total_count_before"] / (
        (features_before["last_purchase_before"] - features_before["first_purchase_before"]).dt.days + 1
    )
    features_before = features_before[["avg_money_before", "avg_count_before"]]

    labels_after = purchases_after.groupby("transaction_id").agg(
        {"client_id": "first", "purchase_sum": "first", "transaction_datetime": "first"}
    ).reset_index()
    labels_after.columns = ["transaction_id", "client_id", "purchase_sum", "transaction_datetime"]
    labels_after = labels_after.groupby("client_id").agg(
        {"purchase_sum": "mean", "transaction_id": "count", "transaction_datetime": ["max", "min"]}
    )
    labels_after.columns = [
        "avg_money_after",
        "total_count_after",
        "last_purchase_after",
        "first_purchase_after",
    ]
    labels_after["avg_count_after"] = labels_after["total_count_after"] / (
        (labels_after["last_purchase_after"] - labels_after["first_purchase_after"]).dt.days + 1
    )
    labels_after = labels_after[["avg_money_after", "avg_count_after"]]

    data = df_features.join(features_before).join(labels_after).fillna(0)
    data["avg_money_change"] = data["avg_money_after"] - data["avg_money_before"]
    data["avg_count_change"] = data["avg_count_after"] - data["avg_count_before"]
    data = data[data.index.isin(purchases["client_id"].unique())].reset_index()

    out = data[
        [
            "age",
            "F",
            "M",
            "U",
            "first_issue_abs_time",
            "first_redeem_abs_time",
            "redeem_delay",
            "avg_money_before",
            "avg_count_before",
        ]
    ].copy()
    out["treatment"] = data["treatment_flg"].astype(np.int8)
    out["outcome"] = data["avg_money_change"].astype(np.float64)
    out.to_csv(retail_csv, index=False)
    return retail_csv


def main():
    root = Path(__file__).resolve().parents[2] / "data" / "retailhero"
    csv_path = prepare_retail_csv(root)
    print(f"Saved: {csv_path.name}")


if __name__ == "__main__":
    main()
