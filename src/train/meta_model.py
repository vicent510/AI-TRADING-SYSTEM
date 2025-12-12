# Libraries imports
import numpy as np
import pandas as pd
import joblib
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb

# Local imports
from src.utils.utils import getTime, bcolors

def meta_model(INPUT_FILE: str, OUTPUT_FILE: str, RANDOM_STATE: int, TEST_SIZE: float):
    print(f"{getTime()}🔄 Starting Meta Model Training...")

    # Load Data
    try:
        meta_data = np.load(INPUT_FILE)
        X_meta = meta_data['X']
        y_meta = meta_data['y']
    except Exception as e:
        print(f"\n{getTime()} {bcolors.FAIL}ERROR: While loading '{INPUT_FILE}'.{bcolors.ENDC}\n")
        sys.exit(1)
    
    # Split
    X_train_meta, X_test_meta, y_train_meta, y_test_meta = train_test_split(X_meta, y_meta, test_size=TEST_SIZE, shuffle=True, stratify=y_meta, random_state=RANDOM_STATE)

    # XGBoost Config
    pos_ratio = y_train_meta.mean()
    scale_pos_weight = (1.0 - pos_ratio) / pos_ratio if pos_ratio > 0 else 1.0

    print(f"{getTime()} ⚖️  Scale Pos Weight: {scale_pos_weight:.2f}")

    # DMatrix
    dtrain = xgb.DMatrix(X_train_meta, label=y_train_meta)
    dtest  = xgb.DMatrix(X_test_meta, label=y_test_meta)

    params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',          
    'max_depth': 5,               
    'eta': 0.03,                   
    'subsample': 0.7,              
    'colsample_bytree': 0.7,      
    'scale_pos_weight': scale_pos_weight,
    'tree_method': 'hist',         
    'seed': RANDOM_STATE
    }

    # Train Model
    print(f"\n{getTime()} 📈 Starting Training...")
    evals = [(dtrain, 'train'), (dtest, 'eval')]

    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,         
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=100
    )

    print(f"{getTime()} Train Set Results")

    y_prob = bst.predict(dtest)
    y_pred = (y_prob > 0.5).astype(int)

    print(classification_report(y_test_meta, y_pred))
    print(f"{getTime()} 🚀 Final ROC AUC: {roc_auc_score(y_test_meta, y_prob):.4f}")

    # Save
    joblib.dump(bst, OUTPUT_FILE)
    print(f"💾 Model Saved: {OUTPUT_FILE}")