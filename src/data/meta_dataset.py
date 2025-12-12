# Dependencies imports
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
import sys

# Local imports
from src.utils.utils import getTime, bcolors

def meta_dataset(INPUT_FILE: str, OUTPUT_FILE: str, MODEL_PATH: str, SEQ_LEN: int, THRESHOLD: float):
    print(f"{getTime()} 🔄 Starting Meta Dataset creation...")

    MODEL_PATH = f"{MODEL_PATH}/best_model.keras"

    # Load Data and Model
    try:
        train_data = np.load(os.path.join(INPUT_FILE, 'train_data.npz'))
        X_train_scaled = train_data['X']
        y_train_true = train_data['y']
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"{getTime()}{bcolors.FAIL}❌ ERROR: While loading sources {bcolors.ENDC}{e}")
        sys.exit()

    # Predictions
    print(f"{getTime()} 🧠 Generating predictions with the main model...")
    lstm_dataset = tf.keras.utils.timeseries_dataset_from_array(
        data=X_train_scaled,
        targets=y_train_true,
        sequence_length=SEQ_LEN,
        sequence_stride=1,
        shuffle=False,
        batch_size=4096 
    )

    preds = model.predict(lstm_dataset, verbose=1) 
    preds = preds.flatten() 

    # Data Lineup
    start_index = SEQ_LEN - 1
    aligned_X_scaled = X_train_scaled[start_index:] 
    aligned_y_true = y_train_true[start_index:]
    min_len = min(len(preds), len(aligned_y_true))

    preds = preds[:min_len]
    aligned_X_scaled = aligned_X_scaled[:min_len]
    aligned_y_true = aligned_y_true[:min_len]

    print(f"{getTime()}{bcolors.OKGREEN} ✅ Data Lined Up: {min_len} samples.{bcolors.ENDC}")

    meta_target = (aligned_y_true >= THRESHOLD).astype(int) 

    # Model Predictions
    meta_features_df = pd.DataFrame()
    meta_features_df['pred_reg'] = preds 
    meta_features_df['confidence_abs'] = np.abs(preds) 
    meta_features_df['confidence_pos'] = preds.clip(min=0) # LONG
    meta_features_df['confidence_neg'] = preds.clip(max=0) # SHORT

    meta_features_df['feat_rsi'] = aligned_X_scaled[:, 1]
    meta_features_df['feat_atr'] = aligned_X_scaled[:, 2]
    meta_features_df['feat_bb_width'] = aligned_X_scaled[:, 4]

    X_meta = meta_features_df.values
    y_meta = meta_target

    np.savez_compressed(OUTPUT_FILE, X=X_meta, y=y_meta)

    print(f"{getTime()}{bcolors.OKGREEN} ✅ Meta Dataset created succesfully.")
    print(f"Shape X_meta: {X_meta.shape}")
    print(f"Shape y_meta: {y_meta.shape}")
    print(f"Meta Target Distribution:")
    print(pd.Series(y_meta).value_counts(normalize=True) * 100)