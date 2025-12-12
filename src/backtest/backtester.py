import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import sys
import os
import json

from src.utils.utils import getTime, bcolors

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs): return iterable

try:
    import xgboost as xgb
except ImportError:
    xgb = None

def load_config(CONFIG_FILE: str):
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            params = json.load(f)
        print(f"{getTime()}{bcolors.OKGREEN} ✅ Configuration loaded successfully: {CONFIG_FILE}{bcolors.ENDC}")
        ENTRY_THRESHOLD        = params['entry_th']
        META_THRESHOLD         = params['meta_th']
        DYNAMIC_EXIT_THRESHOLD = params['dyn_exit']
        RR_TP                  = params['rr_tp']
        RR_SL                  = params['rr_sl']
    else:
        print(f"{getTime()}{bcolors.WARNING} ⚠️ Can't find config file. Using default settings.{bcolors.ENDC}")
        ENTRY_THRESHOLD        = 0.25
        META_THRESHOLD         = 0.60
        DYNAMIC_EXIT_THRESHOLD = 0.02
        RR_TP                  = 2.0
        RR_SL                  = 1.0

    return ENTRY_THRESHOLD, META_THRESHOLD, DYNAMIC_EXIT_THRESHOLD, RR_TP, RR_SL

def run_simulation_logic(npz_path, csv_path, model_lstm, model_meta, desc, 
                         SEQ_LEN, ENTRY_THRESHOLD, META_THRESHOLD, DYNAMIC_EXIT_THRESHOLD,
                         RR_TP, RR_SL):
    try:
        data = np.load(npz_path)
        X_scaled = data['X']
        df_raw = pd.read_csv(csv_path, index_col='time', parse_dates=['time'])
        df_raw.columns = [c.lower() for c in df_raw.columns]
    except Exception as e:
        print(f"{getTime()}{bcolors.FAIL} ❌ Error loading data: {csv_path} -> {bcolors.ENDC}{e}{bcolors.ENDC}")
        return pd.DataFrame()

    required_cols = ['open', 'high', 'low', 'close']
    missing = [c for c in required_cols if c not in df_raw.columns]
    if missing:
        print(f"{getTime()}{bcolors.FAIL} ❌ ERROR: The file {csv_path} is missing required columns: {missing}{bcolors.ENDC}")
        return pd.DataFrame()

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

    min_len      = min(len(lstm_preds), len(df_sim), X_scaled.shape[0] - SEQ_LEN)
    lstm_preds   = lstm_preds[:min_len]
    df_sim       = df_sim.iloc[:min_len]
    X_meta_input = X_scaled[SEQ_LEN: SEQ_LEN + min_len]

    df_features = pd.DataFrame(index=df_sim.index)
    df_features['pred_reg']       = lstm_preds
    df_features['confidence_abs'] = np.abs(lstm_preds)
    df_features['confidence_pos'] = np.clip(lstm_preds, a_min=0.0, a_max=None)
    df_features['confidence_neg'] = np.clip(lstm_preds, a_max=0.0, a_min=None)

    if X_meta_input.shape[1] < 3:
        raise ValueError(
            f"{getTime()}{bcolors.FAIL} X_meta_input has only {X_meta_input.shape[1]} features, "
            f"but the meta-model expects at least 3 extra features.{bcolors.ENDC}"
        )

    df_features['feat_0'] = X_meta_input[:, 0]
    df_features['feat_1'] = X_meta_input[:, 1]
    df_features['feat_2'] = X_meta_input[:, 2]

    try:
        if hasattr(model_meta, "predict_proba"):
            meta_probs = model_meta.predict_proba(df_features.values)[:, 1]
        else:
            if xgb is None:
                raise ImportError(
                    f"{getTime()}{bcolors.FAIL} xgboost is not installed but the meta-model is a Booster. "
                    f"Install xgboost or save the model as an XGBClassifier.{bcolors.ENDC}"
                )
            dmatrix = xgb.DMatrix(df_features.values)
            meta_probs = model_meta.predict(dmatrix)

        meta_probs = np.asarray(meta_probs).flatten()
        if len(meta_probs) != len(df_features):
            raise RuntimeError(
                f"{getTime()}{bcolors.FAIL} meta_probs length mismatch (meta_probs={len(meta_probs)} "
                f"rows={len(df_features)}).{bcolors.ENDC}"
            )
    except Exception as e:
        raise RuntimeError(
            f"{getTime()}{bcolors.FAIL} Error during meta-model prediction. Detail: {bcolors.ENDC}{e}{bcolors.ENDC}"
        )

    if 'atr_val' not in df_sim.columns:
        try:
            import pandas_ta as ta
        except ImportError:
            raise ImportError(
                f"{getTime()}{bcolors.FAIL} pandas_ta is not installed and 'atr_val' is missing. "
                f"Install pandas_ta or add atr_val to the raw CSV.{bcolors.ENDC}"
            )
        df_sim['atr_val'] = ta.atr(df_sim['high'], df_sim['low'], df_sim['close'], length=14)
        df_sim['atr_val'] = df_sim['atr_val'].bfill()

    highs  = df_sim['high'].values
    lows   = df_sim['low'].values
    closes = df_sim['close'].values
    atrs   = df_sim['atr_val'].values
    preds  = lstm_preds
    metas  = meta_probs

    trades = []
    in_pos = False
    pos_type = 0
    entry_p = 0.0
    tp_p = 0.0
    sl_p = 0.0

    for i in tqdm(range(len(df_sim)), desc=desc):
        if in_pos:
            res = 0.0
            reason = ""

            if pos_type == 1:
                hit_tp = highs[i] >= tp_p
                hit_sl = lows[i] <= sl_p
            else:
                hit_tp = lows[i] <= tp_p
                hit_sl = highs[i] >= sl_p

            if hit_tp and hit_sl:
                res = -RR_SL
                reason = "SL_BOTH"
            elif hit_tp:
                res = RR_TP
                reason = "TP"
            elif hit_sl:
                res = -RR_SL
                reason = "SL"
            else:
                if pos_type == 1:
                    if preds[i] < DYNAMIC_EXIT_THRESHOLD:
                        res = (closes[i] - entry_p) / (atrs[i] * RR_SL)
                        reason = "DYN"
                else:
                    if preds[i] > -DYNAMIC_EXIT_THRESHOLD:
                        res = (entry_p - closes[i]) / (atrs[i] * RR_SL)
                        reason = "DYN"

            if res != 0.0:
                trades.append({
                    'entry_time': df_sim.index[i],
                    'pnl_r': float(res),
                    'reason': reason,
                    'direction': int(pos_type)
                })
                in_pos = False

            continue

        want_long  = preds[i] >  ENTRY_THRESHOLD
        want_short = preds[i] < -ENTRY_THRESHOLD
        safe_setup = metas[i] > META_THRESHOLD

        if (want_long or want_short) and safe_setup:
            in_pos = True
            pos_type = 1 if want_long else -1
            entry_p = closes[i]
            atr = atrs[i]

            if pos_type == 1:
                tp_p = entry_p + (atr * RR_TP)
                sl_p = entry_p - (atr * RR_SL)
            else:
                tp_p = entry_p - (atr * RR_TP)
                sl_p = entry_p + (atr * RR_SL)

    return pd.DataFrame(trades)

def compute_summary(df_trades, INITIAL_CAPITAL, RISK_PER_TRADE):
    if df_trades is None or df_trades.empty:
        return {
            "final_capital": float(INITIAL_CAPITAL),
            "net_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "total_trades": 0
        }

    capital = INITIAL_CAPITAL
    equity = [INITIAL_CAPITAL]
    drawdowns = [0.0]
    peak = INITIAL_CAPITAL

    for r_res in df_trades['pnl_r']:
        risk_amt = capital * RISK_PER_TRADE
        pnl_money = risk_amt * r_res
        capital += pnl_money

        peak = max(peak, capital)
        dd = ((capital - peak) / peak * 100.0) if peak > 0 else 0.0
        equity.append(capital)
        drawdowns.append(dd)

    total_ret = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100.0
    max_dd = min(drawdowns) if drawdowns else 0.0
    wins = (df_trades['pnl_r'] > 0).sum()
    win_rate = (wins / len(df_trades)) * 100.0 if len(df_trades) > 0 else 0.0

    gross_profit = df_trades[df_trades['pnl_r'] > 0]['pnl_r'].sum()
    gross_loss = -df_trades[df_trades['pnl_r'] < 0]['pnl_r'].sum()
    pf = (gross_profit / gross_loss) if gross_loss != 0 else 0.0

    return {
        "final_capital": float(capital),
        "net_return_pct": float(total_ret),
        "max_drawdown_pct": float(max_dd),
        "win_rate_pct": float(win_rate),
        "profit_factor": float(pf),
        "total_trades": int(len(df_trades))
    }

def plot_equity_drawdown(df_trades, title_prefix, INITIAL_CAPITAL, RISK_PER_TRADE, save_path):
    if df_trades.empty:
        print(f"{getTime()}{bcolors.WARNING} ⚠️ {title_prefix}: No trades executed.{bcolors.ENDC}")
        return None

    capital = INITIAL_CAPITAL
    equity = [INITIAL_CAPITAL]
    drawdowns = [0.0]
    peak = INITIAL_CAPITAL

    for r_res in df_trades['pnl_r']:
        risk_amt = capital * RISK_PER_TRADE
        pnl_money = risk_amt * r_res
        capital += pnl_money

        peak = max(peak, capital)
        dd = ((capital - peak) / peak * 100.0) if peak > 0 else 0.0

        equity.append(capital)
        drawdowns.append(dd)

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(equity, label='Equity')
    plt.title(f"{title_prefix} - Final: {capital:.0f}€")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(drawdowns, label='Drawdown')
    plt.fill_between(range(len(drawdowns)), drawdowns, 0, alpha=0.3)
    plt.title("Drawdown (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"{getTime()}{bcolors.OKGREEN} ✅ Chart saved: {bcolors.ENDC}{save_path}")
    return compute_summary(df_trades, INITIAL_CAPITAL, RISK_PER_TRADE)

def write_report(report_path, test_summary, hist_summary, params):
    out_dir = os.path.dirname(report_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    lines = []
    lines.append("# Backtest Summary\n")
    lines.append(f"- Generated at: {getTime()}\n")

    lines.append("## Parameters\n")
    lines.append(f"- Entry Threshold: {params['ENTRY_THRESHOLD']}\n")
    lines.append(f"- Meta Threshold: {params['META_THRESHOLD']}\n")
    lines.append(f"- Dynamic Exit Threshold: {params['DYNAMIC_EXIT_THRESHOLD']}\n")
    lines.append(f"- RR TP: {params['RR_TP']}\n")
    lines.append(f"- RR SL: {params['RR_SL']}\n")

    lines.append("\n## Test Results\n")
    lines.append(f"- Final Capital: {test_summary['final_capital']:.2f}\n")
    lines.append(f"- Net Return (%): {test_summary['net_return_pct']:.2f}\n")
    lines.append(f"- Max Drawdown (%): {test_summary['max_drawdown_pct']:.2f}\n")
    lines.append(f"- Win Rate (%): {test_summary['win_rate_pct']:.2f}\n")
    lines.append(f"- Profit Factor: {test_summary['profit_factor']:.2f}\n")
    lines.append(f"- Total Trades: {test_summary['total_trades']}\n")

    lines.append("\n## Full History Results\n")
    lines.append(f"- Final Capital: {hist_summary['final_capital']:.2f}\n")
    lines.append(f"- Net Return (%): {hist_summary['net_return_pct']:.2f}\n")
    lines.append(f"- Max Drawdown (%): {hist_summary['max_drawdown_pct']:.2f}\n")
    lines.append(f"- Win Rate (%): {hist_summary['win_rate_pct']:.2f}\n")
    lines.append(f"- Profit Factor: {hist_summary['profit_factor']:.2f}\n")
    lines.append(f"- Total Trades: {hist_summary['total_trades']}\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"{getTime()}{bcolors.OKGREEN} ✅ Report saved: {bcolors.ENDC}{report_path}")

def backtester(LSTM_PATH: str, META_MODEL_PATH: str, CONFIG_FILE: str, 
               TEST_DATA_NPZ: str, TEST_RAW_CSV: str, TRAIN_DATA_NPZ: str,
               TRAIN_RAW_CSV: str, SEQ_LEN: int, INITIAL_CAPITAL: float, RISK_PER_TRADE: float, OUTPUT_FILE: str):

    print(f"{getTime()} 🔄 Starting Backtest...")
    print(f"{getTime()} 🔍 Searching optimized config...")

    LSTM_PATH = f"{LSTM_PATH}/best_model.keras"
    TEST_DATA_NPZ = f"{TEST_DATA_NPZ}/test_data.npz"
    TEST_RAW_CSV = f"{TEST_RAW_CSV}/test_raw.csv"
    TRAIN_DATA_NPZ = f"{TRAIN_DATA_NPZ}/train_data.npz"
    TRAIN_RAW_CSV = f"{TRAIN_RAW_CSV}/train_raw.csv"

    os.makedirs(OUTPUT_FILE, exist_ok=True)

    report_path = f"{OUTPUT_FILE}/Backtest.md"
    test_chart_path = f"{OUTPUT_FILE}/Test_Chart.jpg"
    hist_chart_path = f"{OUTPUT_FILE}/Historical_Chart.jpg"

    ENTRY_THRESHOLD, META_THRESHOLD, DYNAMIC_EXIT_THRESHOLD, RR_TP, RR_SL = load_config(CONFIG_FILE)
    print(f"{getTime()} ⚙️ Parameters: Entry>{ENTRY_THRESHOLD:.2f} | Meta>{META_THRESHOLD:.2f} | TP:{RR_TP:.2f}R | SL:{RR_SL:.2f}R")

    if not os.path.exists(LSTM_PATH):
        sys.exit(f"{getTime()}{bcolors.FAIL} ❌ Can't find LSTM model{bcolors.ENDC}")

    if not os.path.exists(META_MODEL_PATH):
        sys.exit(f"{getTime()}{bcolors.FAIL} ❌ Can't find META model{bcolors.ENDC}")

    model_lstm = tf.keras.models.load_model(LSTM_PATH)
    model_meta = joblib.load(META_MODEL_PATH)

    print(f"{getTime()} Backtesting Test Dataset...")
    df_test = run_simulation_logic(
        TEST_DATA_NPZ, TEST_RAW_CSV, model_lstm, model_meta, "Test Dataset",
        SEQ_LEN, ENTRY_THRESHOLD, META_THRESHOLD, DYNAMIC_EXIT_THRESHOLD, RR_TP, RR_SL
    )
    test_summary = plot_equity_drawdown(df_test, "Test Dataset", INITIAL_CAPITAL, RISK_PER_TRADE, test_chart_path)
    if test_summary is None:
        test_summary = compute_summary(pd.DataFrame(), INITIAL_CAPITAL, RISK_PER_TRADE)

    print(f"{getTime()} Backtesting Train Dataset...")
    df_train = run_simulation_logic(
        TRAIN_DATA_NPZ, TRAIN_RAW_CSV, model_lstm, model_meta, "Train Dataset",
        SEQ_LEN, ENTRY_THRESHOLD, META_THRESHOLD, DYNAMIC_EXIT_THRESHOLD, RR_TP, RR_SL
    )

    print(f"{getTime()} Test + Train Dataset Backtest...")
    if not df_train.empty and not df_test.empty:
        df_total = pd.concat([df_train, df_test], ignore_index=True)
        df_total.sort_values('entry_time', inplace=True)
        hist_summary = plot_equity_drawdown(df_total, "FULL HISTORY", INITIAL_CAPITAL, RISK_PER_TRADE, hist_chart_path)
        if hist_summary is None:
            hist_summary = compute_summary(pd.DataFrame(), INITIAL_CAPITAL, RISK_PER_TRADE)
    else:
        print(f"{getTime()}{bcolors.WARNING} ⚠️ There's not enough data to run the historical backtest.{bcolors.ENDC}")
        hist_summary = compute_summary(pd.DataFrame(), INITIAL_CAPITAL, RISK_PER_TRADE)

    params = {
        "ENTRY_THRESHOLD": ENTRY_THRESHOLD,
        "META_THRESHOLD": META_THRESHOLD,
        "DYNAMIC_EXIT_THRESHOLD": DYNAMIC_EXIT_THRESHOLD,
        "RR_TP": RR_TP,
        "RR_SL": RR_SL
    }

    write_report(report_path, test_summary, hist_summary, params)
