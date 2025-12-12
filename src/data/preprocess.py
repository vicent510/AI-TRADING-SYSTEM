import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer
import joblib
import os
import sys

from src.utils.utils import getTime, bcolors

EXCLUDE_COLS = [
    'open', 'high', 'low', 'close', 'atr_val',
    'target', 'target_class', 'target_R', 'meta_y', 'future_close'
]

def preprocess(INPUT_FILE: str, OUTPUT_FILE: str, TRAIN_RATIO: float, VAL_RATIO: float):
    print(f"\n{getTime()} 🔄 Starting Data Preprocessing...")
    os.makedirs(OUTPUT_FILE, exist_ok=True)

    if not os.path.isdir(INPUT_FILE):
        print(f"\n{getTime()} {bcolors.FAIL}ERROR: Input directory '{INPUT_FILE}' does not exist.{bcolors.ENDC}\n")
        sys.exit(1)

    train_path = os.path.join(INPUT_FILE, "train_raw.csv")
    val_path   = os.path.join(INPUT_FILE, "val_raw.csv")
    test_path  = os.path.join(INPUT_FILE, "test_raw.csv")

    for p in [train_path, val_path, test_path]:
        if not os.path.exists(p):
            print(f"\n{getTime()} {bcolors.FAIL}ERROR: Required file '{p}' does not exist.{bcolors.ENDC}\n")
            sys.exit(1)

    print(f"{getTime()} 📂 Loading datasets from: {INPUT_FILE}")
    train_df = pd.read_csv(train_path, index_col='time', parse_dates=['time'])
    val_df   = pd.read_csv(val_path,   index_col='time', parse_dates=['time'])
    test_df  = pd.read_csv(test_path,  index_col='time', parse_dates=['time'])

    espaces = " " * 22
    print(f"{getTime()}{bcolors.OKGREEN} ✅ Data Loaded:")
    print(f"{espaces}Train: {len(train_df)} rows")
    print(f"{espaces}Val:   {len(val_df)} rows")
    print(f"{espaces}Test:  {len(test_df)} rows")

    FEATURES = [c for c in train_df.columns if c not in EXCLUDE_COLS]

    print(f"{getTime()}📝 DETECTED FEATURES ({len(FEATURES)}): {FEATURES[:5]} ...")

    X_train = train_df[FEATURES].values
    y_train = train_df['target'].values
    y_meta_train = train_df['meta_y'].values

    X_val = val_df[FEATURES].values
    y_val = val_df['target'].values
    y_meta_val = val_df['meta_y'].values

    X_test = test_df[FEATURES].values
    y_test = test_df['target'].values
    y_meta_test = test_df['meta_y'].values

    scaler = QuantileTransformer(output_distribution='normal', random_state=42, n_quantiles=2000)

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    joblib.dump(scaler, os.path.join(OUTPUT_FILE, 'scaler.pkl'))

    print(f"{getTime()} 💾 Saving processed dataset (.npz)...")

    np.savez_compressed(
        os.path.join(OUTPUT_FILE, 'train_data.npz'),
        X=X_train_scaled, y=y_train, y_meta=y_meta_train
    )

    np.savez_compressed(
        os.path.join(OUTPUT_FILE, 'val_data.npz'),
        X=X_val_scaled, y=y_val, y_meta=y_meta_val
    )

    np.savez_compressed(
        os.path.join(OUTPUT_FILE, 'test_data.npz'),
        X=X_test_scaled, y=y_test, y_meta=y_meta_test
    )

    print(f"{getTime()}{bcolors.OKGREEN} ✅ Preprocessing completed. Saving into: {OUTPUT_FILE}")
