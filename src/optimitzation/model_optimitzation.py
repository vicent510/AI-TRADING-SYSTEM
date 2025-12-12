# Dependencies imports
import optuna
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import xgboost as xgb 
import os
import sys
import json

# Local imports
from src.utils.utils import getTime, bcolors

def build_data_dict(npz_path, raw_csv_path, SEQ_LEN, model_lstm, model_meta):
    data_npz = np.load(npz_path)
    X_scaled = data_npz['X']

    df_raw = pd.read_csv(raw_csv_path, index_col='time', parse_dates=['time'])

    dataset = tf.keras.utils.timeseries_dataset_from_array(
        data=X_scaled,
        targets=None,
        sequence_length=SEQ_LEN,
        sequence_stride=1,
        shuffle=False,
        batch_size=4096
    )
    lstm_preds = model_lstm.predict(dataset, verbose=0).flatten()

    sim_len = len(lstm_preds)
    df_sim = df_raw.iloc[SEQ_LEN: SEQ_LEN + sim_len].copy()
    min_len = min(len(lstm_preds), len(df_sim))

    lstm_preds = lstm_preds[:min_len]
    df_sim     = df_sim.iloc[:min_len]
    X_meta_inp = X_scaled[SEQ_LEN:][:min_len]

    df_features = pd.DataFrame()
    df_features['pred_reg']       = lstm_preds
    df_features['confidence_abs'] = np.abs(lstm_preds)
    df_features['confidence_pos'] = lstm_preds.clip(min=0)
    df_features['confidence_neg'] = lstm_preds.clip(max=0)
    df_features['feat_rsi']       = X_meta_inp[:, 1]
    df_features['feat_atr']       = X_meta_inp[:, 2]
    df_features['feat_bb_width']  = X_meta_inp[:, 4]

    dmeta = xgb.DMatrix(df_features.values)
    meta_probs = model_meta.predict(dmeta)

    if 'atr_val' not in df_sim.columns:
        import pandas_ta as ta
        df_sim['atr_val'] = ta.atr(df_sim['high'], df_sim['low'], df_sim['close'], length=14)
        df_sim['atr_val'].fillna(method='bfill', inplace=True)

    data_dict = {
        'close':  df_sim['close'].values,
        'high':   df_sim['high'].values,
        'low':    df_sim['low'].values,
        'atr':    df_sim['atr_val'].values,
        'pred_r': lstm_preds,
        'meta_p': meta_probs
    }

    operable_days = df_sim.index.normalize().nunique()

    return data_dict, operable_days

def simulate_and_score(params, data_dict, operable_days, INITIAL_CAPITAL, RISK_PER_TRADE, N_MIN_TRADES, weights, allow_negative=False):
    entry_th = params['entry_th']
    meta_th  = params['meta_th']
    dyn_exit = params['dyn_exit']
    rr_tp    = params['rr_tp']
    rr_sl    = params['rr_sl']

    closes = data_dict['close']
    highs  = data_dict['high']
    lows   = data_dict['low']
    atrs   = data_dict['atr']
    preds  = data_dict['pred_r']
    metas  = data_dict['meta_p']

    capital     = INITIAL_CAPITAL
    peak        = INITIAL_CAPITAL
    max_dd_pct  = 0.0
    trades_count= 0
    wins        = 0
    gross_profit_r = 0.0
    gross_loss_r   = 0.0
    r_per_trade    = []

    in_pos  = False
    pos_type= 0
    entry_p = 0.0
    tp_p    = 0.0
    sl_p    = 0.0

    for i in range(len(closes)):
        # Open Position
        if in_pos:
            res = 0.0

            if pos_type == 1:  # LONG
                if highs[i] >= tp_p:
                    res = rr_tp
                elif lows[i] <= sl_p:
                    res = -rr_sl
                elif preds[i] < dyn_exit:
                    res = (closes[i] - entry_p) / (atrs[i] * rr_sl)
            else:              # SHORT
                if lows[i] <= tp_p:
                    res = rr_tp
                elif highs[i] >= sl_p:
                    res = -rr_sl
                elif preds[i] > -dyn_exit:
                    res = (entry_p - closes[i]) / (atrs[i] * rr_sl)

            if res != 0.0:
                pnl = capital * RISK_PER_TRADE * res
                capital += pnl
                peak = max(peak, capital)
                if peak > 0:
                    dd = (capital - peak) / peak
                    if dd < max_dd_pct:
                        max_dd_pct = dd

                trades_count += 1
                r_per_trade.append(res)

                if res > 0:
                    wins           += 1
                    gross_profit_r += res
                else:
                    gross_loss_r   += res

                in_pos = False

                if capital < INITIAL_CAPITAL * 0.5:
                    return -1e9, {'net_return': -1, 'DD': abs(max_dd_pct), 'PF': 0.0, 'WR': 0.0, 'trades': trades_count}
            continue

        # Search for entry
        go_long  = preds[i] >  entry_th
        go_short = preds[i] < -entry_th
        safe     = metas[i] >  meta_th

        if not in_pos and (go_long or go_short) and safe:
            in_pos  = True
            pos_type= 1 if go_long else -1
            entry_p = closes[i]
            atr     = atrs[i]

            if pos_type == 1:  # LONG
                tp_p = entry_p + (atr * rr_tp)
                sl_p = entry_p - (atr * rr_sl)
            else:              # SHORT
                tp_p = entry_p - (atr * rr_tp)
                sl_p = entry_p + (atr * rr_sl)

    if trades_count < N_MIN_TRADES:
        return -1e9, {'net_return': 0.0, 'DD': abs(max_dd_pct), 'PF': 0.0, 'WR': 0.0, 'trades': trades_count}

    net_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
    if (not allow_negative) and (net_return <= 0):
        return -1e9, {'net_return': net_return, 'DD': abs(max_dd_pct), 'PF': 0.0, 'WR': 0.0, 'trades': trades_count}

    DD = abs(max_dd_pct)

    if gross_loss_r < 0:
        PF = gross_profit_r / abs(gross_loss_r)
    else:
        PF = 10.0

    WR = wins / trades_count if trades_count > 0 else 0.0
    trades_per_day = trades_count / max(operable_days, 1)

    r_arr   = np.array(r_per_trade, dtype=float)
    mu_R    = np.mean(r_arr)
    sigma_R = np.std(r_arr) + 1e-9
    Stab    = mu_R / sigma_R
    Stab_clipped = np.clip(Stab, -2.0, 5.0)

    # TERMS SCORE
    Profit_term = np.log1p(max(net_return, 0.0))
    PF_term     = np.log1p(max(PF - 1.0, 0.0))
    WR_term     = 2.0 * (WR - 0.5)
    Act_term    = np.log1p(trades_per_day)

    score = (
        weights["W_R"]     * Profit_term  +
        weights["W_PF"]    * PF_term      +
        weights["W_WR"]    * WR_term      +
        weights["W_Stab"]  * Stab_clipped +
        weights["W_Act"]   * Act_term     -
        weights["W_DD"]    * DD
    )

    metrics = {
        'net_return': net_return, 'DD': DD, 'PF': PF, 'WR': WR,
        'trades': trades_count, 'Stab': Stab_clipped, 'score': score
    }
    return score, metrics


def model_optimitzation(LSTM_PATH: str, META_MODEL_PATH: str, TRAIN_DATA_NPZ: str, TRAIN_RAW_CSV: str, VAL_DATA_NPZ: str, VAL_RAW_CSV: str, CONFIG_FILE: str, N_TRIALS: int, INITIAL_CAPITAL: float, RISK_PER_TRADE: float, SEQ_LEN: int, N_MIN_TRADES: int, weights: dict):
    print(f"\n{getTime()} 🔄 Starting Model Optimitzation...")

    LSTM_PATH = f"{LSTM_PATH}/best_model.keras"
    TRAIN_DATA_NPZ = f"{TRAIN_DATA_NPZ}/train_data.npz"
    TRAIN_RAW_CSV = f"{TRAIN_RAW_CSV}/train_raw.csv"
    VAL_DATA_NPZ = f"{VAL_DATA_NPZ}/val_data.npz"
    VAL_RAW_CSV = f"{VAL_RAW_CSV}/val_raw.csv"

    if not os.path.exists(LSTM_PATH) or not os.path.exists(META_MODEL_PATH):
        print(f"{bcolors.FAIL} ❌ ERROR: Missing Models.")
        sys.exit()
    
    model_lstm = tf.keras.models.load_model(LSTM_PATH)
    model_meta = joblib.load(META_MODEL_PATH)

    print(f"{getTime()} Preparing dataset for TRAIN optimitzation...")
    DATA_TRAIN, OPERABLE_DAYS_TRAIN = build_data_dict(TRAIN_DATA_NPZ, TRAIN_RAW_CSV, SEQ_LEN, model_lstm, model_meta)

    print(f"{getTime()} Preparing dataset for TRAIN + VALIDATION verification...")

    train_npz = np.load(TRAIN_DATA_NPZ)
    val_npz   = np.load(VAL_DATA_NPZ)
    X_all     = np.concatenate([train_npz['X'], val_npz['X']], axis=0)

    df_train_raw = pd.read_csv(TRAIN_RAW_CSV, index_col='time', parse_dates=['time'])
    df_val_raw   = pd.read_csv(VAL_RAW_CSV,   index_col='time', parse_dates=['time'])
    df_all_raw   = pd.concat([df_train_raw, df_val_raw]).sort_index()

    dataset_all = tf.keras.utils.timeseries_dataset_from_array(
        data=X_all,
        targets=None,
        sequence_length=SEQ_LEN,
        sequence_stride=1,
        shuffle=False,
        batch_size=4096
    )

    lstm_preds_all = model_lstm.predict(dataset_all, verbose=0).flatten()

    sim_len_all = len(lstm_preds_all)
    df_all_sim  = df_all_raw.iloc[SEQ_LEN: SEQ_LEN + sim_len_all].copy()
    min_len_all = min(len(lstm_preds_all), len(df_all_sim))

    lstm_preds_all = lstm_preds_all[:min_len_all]
    df_all_sim     = df_all_sim.iloc[:min_len_all]
    X_meta_all     = X_all[SEQ_LEN:][:min_len_all]

    df_features_all = pd.DataFrame()
    df_features_all['pred_reg']       = lstm_preds_all
    df_features_all['confidence_abs'] = np.abs(lstm_preds_all)
    df_features_all['confidence_pos'] = lstm_preds_all.clip(min=0)
    df_features_all['confidence_neg'] = lstm_preds_all.clip(max=0)
    df_features_all['feat_rsi']       = X_meta_all[:, 1]
    df_features_all['feat_atr']       = X_meta_all[:, 2]
    df_features_all['feat_bb_width']  = X_meta_all[:, 4]

    dmeta_all = xgb.DMatrix(df_features_all.values)
    meta_probs_all = model_meta.predict(dmeta_all)

    if 'atr_val' not in df_all_sim.columns:
        import pandas_ta as ta
        df_all_sim['atr_val'] = ta.atr(df_all_sim['high'], df_all_sim['low'], df_all_sim['close'], length=14)
        df_all_sim['atr_val'].fillna(method='bfill', inplace=True)

    DATA_ALL = {
        'close':  df_all_sim['close'].values,
        'high':   df_all_sim['high'].values,
        'low':    df_all_sim['low'].values,
        'atr':    df_all_sim['atr_val'].values,
        'pred_r': lstm_preds_all,
        'meta_p': meta_probs_all
    }
    OPERABLE_DAYS_ALL = df_all_sim.index.normalize().nunique()

    def objective(trial):
        params = {
            'entry_th': trial.suggest_float('entry_th', 0.15, 0.50, step=0.01),
            'meta_th':  trial.suggest_float('meta_th',  0.50, 0.95, step=0.01),
            'dyn_exit': trial.suggest_float('dyn_exit', 0.0,  0.10, step=0.01),
            'rr_tp':    trial.suggest_float('rr_tp',    2.0,  5.0,  step=0.1), 
            'rr_sl':    trial.suggest_float('rr_sl',    0.8,  1.5,  step=0.1),
        }

        score, _ = simulate_and_score(params, DATA_TRAIN, OPERABLE_DAYS_TRAIN, INITIAL_CAPITAL, RISK_PER_TRADE, N_MIN_TRADES, weights, True)
        return score

    # Execute Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=N_TRIALS)

    print(f"{getTime()} Optimitzation Finished")
    best_params = study.best_params
    print(f"{getTime()} Best Parameters: {best_params}")

    # TRAIN + VALIDATION Verification
    print(f"{getTime()} Verifying Best Params in TRAIN+VAL dataset...")
    best_score_all, metrics_all = simulate_and_score(best_params, DATA_ALL, OPERABLE_DAYS_ALL, INITIAL_CAPITAL, RISK_PER_TRADE, N_MIN_TRADES, weights, True)

    print(f"{getTime()} {bcolors.OKGREEN}📊 RESULTS TRAIN+VAL:")
    print(f"   Score         : {metrics_all['score']:.4f}")
    print(f"   Net Return    : {metrics_all['net_return']*100:.2f}%")
    print(f"   Max DD        : {metrics_all['DD']*100:.2f}%")
    print(f"   Profit Factor : {metrics_all['PF']:.2f}")
    print(f"   WinRate       : {metrics_all['WR']*100:.2f}%")
    print(f"   Trades        : {metrics_all['trades']}")
    print(f"   Stab (μ/σ R)  : {metrics_all['Stab']:.3f}{bcolors.ENDC}")

    # Save Config
    with open(CONFIG_FILE, 'w') as f:
        json.dump(best_params, f, indent=4)

    print(f"{getTime()} 💾 Config saved into: {CONFIG_FILE}")